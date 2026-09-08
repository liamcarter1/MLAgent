"""Codegen stage: copy the training template and choose a starting config."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlagent import config as cfg
from mlagent.llm import LLMError, ToolSpec
from mlagent.prompts_io import audience, load_prompt
from mlagent.spec import Spec
from mlagent.stages.base import StageContext
from mlagent.stages.data import META_FILE
from mlagent.templates_io import (
    CODE_FILES,
    TEMPLATE_FOR_TASK,
    coerce_config,
    copy_template,
    load_schema,
    model_types,
    schema_for,
    validate_config,
)

MAX_LISTED = 20
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


def check_data(meta: dict, project_root: Path) -> list[str]:
    """Problems that would make training impossible; empty list means go ahead."""
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
    splits = meta.get("splits") or {}
    fractions = [float(splits.get(k, 0)) for k in ("train", "val", "test")]
    total = sum(fractions)
    if abs(total - 1.0) > 0.01 or any(f <= 0 for f in fractions):
        problems.append(f"split fractions {splits} must be positive and sum to 1")
    if meta.get("task_type") == "tabular_classification" and (meta.get("n_classes") or 0) < 2:
        problems.append("classification needs at least 2 classes in the target")
    return problems


def meta_summary(meta: dict) -> dict:
    features = list(meta.get("feature_columns") or [])
    labels = list(meta.get("class_labels") or [])
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
    }


def config_table(config: dict, schema: dict) -> str:
    rows = ["| key | value | what it does |", "|---|---|---|"]
    for key, value in config.items():
        desc = schema.get(key, {}).get("description", "")
        if value is None:
            shown = "none"
        elif isinstance(value, float):
            shown = f"{value:g}"
        else:
            shown = str(value)
        rows.append(f"| {key} | {shown} | {desc} |")
    return "\n".join(rows)


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
            schema = load_schema(TEMPLATE_FOR_TASK[spec.task_type])
            flat = schema_for(schema, str(config.get("model_type", "")))
        except Exception:  # noqa: BLE001 - missing spec, unknown model or task: not complete
            return False
        return validate_config(config, flat) == []

    def prepare(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        template = TEMPLATE_FOR_TASK.get(spec.task_type)
        if template is None:
            ctx.display(
                f"No training template for task type `{spec.task_type}` yet; "
                "this milestone covers tabular tasks only."
            )
            return
        meta = ctx.project.read_json(META_FILE) or {}
        problems = check_data(meta, ctx.project.root)
        if problems:
            ctx.display("The data is not ready for training:\n- " + "\n- ".join(problems))
            return

        nested = load_schema(template)
        schema = schema_for(nested, model_types(nested)[0])
        proposal, rationale = self._propose(ctx, spec, meta, schema)
        config, notes = coerce_config(proposal, schema)
        written = copy_template(template, ctx.project.root)
        ctx.project.write_json(cfg.CONFIG_FILE, config)

        files = ", ".join(f"`{p.name}`" for p in written) + ", `config.json`"
        message = [
            f"I wrote the training project into the project folder: {files}.",
            "`train.py` trains [[gradient boosting]] trees; each [[epoch]] adds boosting rounds "
            "and records train and validation [[loss]] so we can watch for [[overfitting]].",
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
            config = self._edit_config(ctx, config, schema)
            ctx.project.write_json(cfg.CONFIG_FILE, config)
            ctx.display("Updated configuration:\n\n" + config_table(config, schema))

    def debrief(self, ctx: StageContext) -> None:
        """Codegen needs no cells from the user; everything happened in prepare."""
        return None

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

    def _edit_config(self, ctx: StageContext, config: dict, schema: dict) -> dict:
        config = dict(config)
        while True:
            # model_type (and any other "choice" rule) is not a number the questioner can
            # prompt for; Task 10 owns the model-choice flow, so it is left off this menu.
            editable = [
                k for k in config if schema.get(k, {}).get("type") != "choice"
            ]
            options = [f"{k} = {config[k]}" for k in editable] + ["Done"]
            pick = ctx.questioner.choice(
                "Which value do you want to change?", options, allow_other=False
            )
            if pick == "Done":
                return config
            key = pick.split(" = ", 1)[0]
            rule = schema.get(key)
            if rule is None:
                continue
            nullable = bool(rule.get("nullable"))
            hint = " (0 means no limit)" if nullable else ""
            current = config.get(key)
            value = ctx.questioner.number(
                f"New value for {key}{hint}: {rule.get('description', '')}",
                default=0 if current is None else current,
                minimum=0 if nullable else rule.get("min"),
                maximum=rule.get("max"),
            )
            if nullable and value == 0:
                config[key] = None
            elif rule.get("type") == "integer":
                config[key] = int(round(value))
            else:
                config[key] = float(value)
            problems = validate_config(config, schema)
            if problems:
                ctx.display("That value is not allowed: " + "; ".join(problems))
                config[key] = current
