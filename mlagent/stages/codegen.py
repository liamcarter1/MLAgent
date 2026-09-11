"""Codegen stage: copy the training template and choose a starting config."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlagent import config as cfg
from mlagent.llm import LLMError, ToolSpec
from mlagent.modality import modality_for
from mlagent.prompts_io import audience, load_prompt
from mlagent.spec import Spec
from mlagent.stages.base import StageContext
from mlagent.stages.data import META_FILE
from mlagent.teaching import material
from mlagent.templates_io import (
    CODE_FILES,
    coerce_config,
    config_table,
    copy_template,
    edit_config,
    load_schema,
    model_types,
    schema_for,
    validate_config,
)

MAX_LISTED = 20
SMALL_DATA_ROWS = 300
MODEL_LABELS_BY_FAMILY: dict[str, dict[str, str]] = {
    "tabular_sklearn": {
        "Linear / logistic regression": "linear",
        "Random forest": "random_forest",
        "Gradient boosting": "gradient_boosting",
    },
    "image_torch": {
        "Tiny CNN": "tiny_cnn",
        "Small CNN": "small_cnn",
        "Pretrained ResNet-18": "resnet18",
    },
}
# Kept as the tabular map so `codegen.MODEL_LABELS` still resolves for older callers.
MODEL_LABELS = MODEL_LABELS_BY_FAMILY["tabular_sklearn"]
FALLBACK_MODEL_BY_FAMILY = {"tabular_sklearn": "gradient_boosting", "image_torch": "small_cnn"}
ASK_LABEL = "Ask me after the explanation"


def labels_for(family: str) -> dict[str, str]:
    """The label -> model_type map for one template family."""
    return MODEL_LABELS_BY_FAMILY[family]


def label_for(family: str, code: str) -> str:
    """The human label for one model_type, or the code itself if it has none."""
    for label, value in MODEL_LABELS_BY_FAMILY.get(family, {}).items():
        if value == code:
            return label
    return code


PROPOSE_TOOL = ToolSpec(
    name="propose_config",
    description=(
        "Propose starting hyperparameters. `config` may only contain keys from the schema; "
        "omit keys left at their default. `rationale` explains the choices to a learner."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {"type": "object", "description": "schema key -> value"},
            "rationale": {"type": "string"},
        },
        "required": ["config", "rationale"],
    },
    handler=lambda inp: "recorded",
)

def recommend_tool(choices: list[str], handler) -> ToolSpec:
    return ToolSpec(
        name="recommend_model",
        description=(
            "Recommend one model family for this dataset. `reason` is shown to the user "
            "before they choose, so make it about their data, not about models in general."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "model_type": {"type": "string", "enum": list(choices)},
                "reason": {"type": "string"},
            },
            "required": ["model_type", "reason"],
        },
        handler=handler,
    )


def fallback_model(meta: dict, family: str = "tabular_sklearn") -> str:
    """What to recommend when Claude is unreachable: flexibility needs rows."""
    default = FALLBACK_MODEL_BY_FAMILY.get(family, "gradient_boosting")
    if family != "tabular_sklearn":
        return default
    try:
        rows = int(meta.get("clean_n_rows") or 0)
    except (TypeError, ValueError):
        return default
    return "linear" if 0 < rows < SMALL_DATA_ROWS else default


def _check_splits(meta: dict) -> list[str]:
    splits = meta.get("splits") or {}
    fractions = [float(splits.get(k, 0)) for k in ("train", "val", "test")]
    if abs(sum(fractions) - 1.0) > 0.01 or any(f <= 0 for f in fractions):
        return [f"split fractions {splits} must be positive and sum to 1"]
    return []


def _check_image_data(meta: dict, project_root: Path) -> list[str]:
    problems: list[str] = []
    clean_path = project_root / str(meta.get("clean_path") or "data/clean/data.npz")
    if not clean_path.exists():
        return [f"clean image file {clean_path.name} is missing; rerun the clean stage"]
    if not (clean_path.parent / "manifest.csv").exists():
        problems.append("manifest.csv is missing next to the clean images")
    if int(meta.get("n_classes") or 0) < 2:
        problems.append("image classification needs at least 2 classes")
    if int(meta.get("clean_n_rows") or 0) < 2:
        problems.append("no images are available for training")
    return problems + _check_splits(meta)


def check_data(meta: dict, project_root: Path) -> list[str]:
    """Problems that would make training impossible; empty list means go ahead."""
    if meta.get("modality") == "image":
        return _check_image_data(meta, project_root)
    problems: list[str] = []
    clean_path = project_root / str(meta.get("clean_path") or "data/clean/data.csv")
    if not clean_path.exists():
        return [f"clean data file {clean_path.name} is missing; rerun the clean stage"]
    columns = list(pd.read_csv(clean_path, nrows=0).columns)
    target = meta.get("target")
    if not target or target not in columns:
        problems.append(f"target column {target!r} is not in the clean data")
    features = [c for c in meta.get("feature_columns") or [] if c in columns and c != target]
    if not features:
        problems.append("no feature columns are available for training")
    problems += _check_splits(meta)
    if meta.get("task_type") == "tabular_classification" and (meta.get("n_classes") or 0) < 2:
        problems.append("classification needs at least 2 classes in the target")
    return problems


def meta_summary(meta: dict) -> dict:
    features = list(meta.get("feature_columns") or [])
    labels = list(meta.get("class_labels") or [])
    modality = meta.get("modality", "tabular")
    return {
        "task_type": meta.get("task_type"),
        "target": meta.get("target"),
        "n_rows": meta.get("clean_n_rows"),
        "n_features": len(features),
        "feature_columns": features[:MAX_LISTED],
        "categorical_columns": list(meta.get("categorical_columns") or [])[:MAX_LISTED],
        "n_classes": meta.get("n_classes"),
        "class_labels": labels[:MAX_LISTED],
        "splits": meta.get("splits"),
        "modality": modality,
        "image_size": meta.get("image_size"),
        "n_images": meta.get("clean_n_rows") if modality == "image" else None,
    }


class CodegenStage:
    name = "codegen"

    def is_complete(self, ctx: StageContext) -> bool:
        root = ctx.project.root
        if not all((root / f).exists() for f in CODE_FILES):
            return False
        config = ctx.project.read_json(cfg.CONFIG_FILE)
        if not isinstance(config, dict):
            return False
        try:
            spec = ctx.spec()
            schema = load_schema(modality_for(spec.task_type).template_family)
            flat = schema_for(schema, str(config.get("model_type", "")))
        except Exception:  # noqa: BLE001 - missing spec, unknown model or task: not complete
            return False
        return validate_config(config, flat) == []

    def prepare(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        modality = modality_for(spec.task_type)
        template = modality.template_family
        meta = ctx.project.read_json(META_FILE) or {}
        problems = check_data(meta, ctx.project.root)
        if problems:
            ctx.display("The data is not ready for training:\n- " + "\n- ".join(problems))
            return None

        nested = load_schema(template)
        model_type = self._choose_model(ctx, spec, meta, nested, modality)
        schema = schema_for(nested, model_type)

        proposal, rationale = self._propose(ctx, spec, meta, schema)
        proposal = {**proposal, "model_type": model_type}
        config, notes = coerce_config(proposal, schema)
        written = copy_template(template, ctx.project.root)
        ctx.project.write_json(cfg.CONFIG_FILE, config)

        files = ", ".join(f"`{p.name}`" for p in written) + ", `config.json`"
        message = [
            f"I wrote the training project into the project folder: {files}.",
            f"`train.py` trains a **{label_for(template, model_type)}** model; each "
            "[[epoch]] is one pass that records train and validation [[loss]] so we can "
            "watch for [[overfitting]]. `evaluate.py` scores one saved model on one split.",
            "",
            rationale,
            "",
            config_table(config, schema),
        ]
        if notes:
            message += ["", "Adjustments to keep values inside the schema:"]
            message += [f"- {n}" for n in notes]
        ctx.display("\n".join(message))

        if not ctx.questioner.confirm("Happy with this configuration? (No lets you change values)"):
            config = edit_config(ctx.questioner, config, schema, ctx.display)
            ctx.project.write_json(cfg.CONFIG_FILE, config)
            ctx.display("Updated configuration:\n\n" + config_table(config, schema))

        self._walkthrough(ctx, written)
        return None

    def debrief(self, ctx: StageContext) -> None:
        """Codegen needs no cells from the user; everything happened in prepare."""
        return None

    def _walkthrough(self, ctx: StageContext, written: list[Path]) -> None:
        ctx.teaching().walkthrough(written)

    def _choose_model(self, ctx, spec, meta: dict, nested: dict, modality) -> str:
        family = modality.template_family
        labels = labels_for(family)
        recommended, reason = self._recommend(ctx, spec, meta, nested, modality)
        ctx.display(material(modality.teaching_material, ctx.learning_level()))
        ctx.display(f"**My recommendation: {label_for(family, recommended)}.** {reason}")
        answer = ctx.questioner.choice(
            f"Which model shall I set up? (I recommend {label_for(family, recommended)})",
            [ASK_LABEL, *labels], allow_other=False, key="codegen.model_type",
        )
        if answer == ASK_LABEL:
            return recommended
        return labels.get(answer, recommended)

    def _recommend(self, ctx, spec, meta: dict, nested: dict, modality) -> tuple[str, str]:
        family = modality.template_family
        choices = model_types(nested)
        captured: dict = {}

        def handler(inp: dict) -> str:
            captured["model_type"] = str(inp.get("model_type") or "")
            captured["reason"] = str(inp.get("reason") or "")
            return "recorded"

        tool = recommend_tool(choices, handler)
        prompt = json.dumps(
            {"spec": spec.to_dict(), "data": meta_summary(meta), "model_choices": choices,
             "task": "recommend one model family"},
            indent=2, default=str,
        )
        try:
            ctx.llm.run(
                load_prompt("codegen", audience=audience(spec.learning_level)),
                [{"role": "user", "content": prompt}], [tool],
            )
        except LLMError as exc:
            chosen = fallback_model(meta, family)
            rows = meta.get("clean_n_rows")
            size = f" ({rows} examples)" if rows is not None else ""
            return chosen, (
                f"(The assistant was unavailable: {exc}.) Going by the size of the dataset"
                f"{size}, {label_for(family, chosen)} is the safe default."
            )
        chosen = captured.get("model_type") or ""
        if chosen not in choices:
            chosen = fallback_model(meta, family)
        return chosen, captured.get("reason") or "It suits the shape of this dataset."

    def _propose(self, ctx: StageContext, spec: Spec, meta: dict, schema: dict) -> tuple[dict, str]:
        captured: dict = {}

        def handler(inp: dict) -> str:
            captured["config"] = inp.get("config") or {}
            captured["rationale"] = str(inp.get("rationale") or "")
            return "recorded"

        tool = ToolSpec(
            name=PROPOSE_TOOL.name,
            description=PROPOSE_TOOL.description,
            input_schema=PROPOSE_TOOL.input_schema,
            handler=handler,
        )
        prompt = json.dumps(
            {"spec": spec.to_dict(), "data": meta_summary(meta), "config_schema": schema},
            indent=2,
            default=str,
        )
        try:
            result = ctx.llm.run(
                load_prompt("codegen", audience=audience(spec.learning_level)),
                [{"role": "user", "content": prompt}],
                [tool],
            )
        except LLMError as exc:
            return {}, f"Using the template defaults (the assistant was unavailable: {exc})."
        rationale = captured.get("rationale") or result.text or "Using the template defaults."
        return dict(captured.get("config") or {}), rationale
