"""Data stage: obtain raw tabular data, profile it, plot it, and narrate what to notice."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlagent import plots
from mlagent.datasources.drive import list_candidates, load_table
from mlagent.datasources.hf import load_tabular, search_datasets
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_dataframe, profile_markdown
from mlagent.prompts_io import audience, load_prompt
from mlagent.stages.base import StageContext
from mlagent.synth.tabular import TARGET, SynthTabularConfig, generate

RAW_FILE = "data.csv"
META_FILE = "data_meta.json"
PROFILE_RAW_FILE = "profile_raw.json"
DEFAULT_QUIRKS = ("missing", "duplicates", "id_column", "categorical", "whitespace", "outliers")
DEFAULT_SEARCH_ROOTS = (Path("/content/drive/MyDrive"), Path("/content"))
TABULAR_TASKS = {"tabular_classification": "classification", "tabular_regression": "regression"}


class DataStage:
    name = "data"

    def __init__(self, search_roots=None, hf_search=search_datasets, hf_load=load_tabular):
        self.search_roots = (
            list(search_roots) if search_roots is not None else list(DEFAULT_SEARCH_ROOTS)
        )
        self.hf_search = hf_search
        self.hf_load = hf_load

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE)
        return bool(meta and meta.get("target")) and (ctx.project.data_raw / RAW_FILE).exists()

    def run(self, ctx: StageContext) -> None:
        spec = ctx.spec()
        if spec.task_type not in TABULAR_TASKS:
            raise NotImplementedError(
                f"{spec.task_type} data is not supported yet (image tasks arrive in Milestone 5)"
            )
        if spec.data_source == "synthetic":
            df, target, meta = self._synthetic(ctx, TABULAR_TASKS[spec.task_type])
        elif spec.data_source == "drive":
            df, target, meta = self._drive(ctx)
        else:
            df, target, meta = self._huggingface(ctx)

        ctx.project.data_raw.mkdir(parents=True, exist_ok=True)
        path = ctx.project.data_raw / RAW_FILE
        df.to_csv(path, index=False)
        profile = profile_dataframe(df, target)
        ctx.project.write_json(PROFILE_RAW_FILE, profile)
        meta.update(
            {
                "target": target,
                "raw_path": path.relative_to(ctx.project.root).as_posix(),
                "raw_n_rows": int(len(df)),
                "raw_n_cols": int(df.shape[1]),
                "task_type": spec.task_type,
            }
        )
        ctx.project.write_json(META_FILE, meta)

        ctx.display(profile_markdown(profile))
        self._plots(ctx, df, profile)
        self._narrate(ctx, spec.to_dict(), profile)

    def _synthetic(self, ctx: StageContext, task: str):
        q = ctx.questioner
        n_samples = int(q.number("How many rows?", default=1000, minimum=100, maximum=200000))
        n_features = int(q.number("How many numeric features?", default=8, minimum=2, maximum=100))
        n_classes, class_balance = 2, 0.5
        if task == "classification":
            n_classes = int(q.number("How many classes?", default=2, minimum=2, maximum=10))
            if n_classes == 2:
                class_balance = q.number(
                    "Fraction of rows in the majority class (0.5 = balanced)?",
                    default=0.5, minimum=0.5, maximum=0.95,
                )
        noise = q.number(
            "Label/measurement noise (0 = clean, 0.3 = very noisy)?",
            default=0.1, minimum=0.0, maximum=1.0,
        )
        inject = q.confirm(
            "Inject realistic data problems (missing values, duplicates, an ID column, messy "
            "categories, outliers) so the cleaning stage has work to do?",
            default=True,
        )
        cfg = SynthTabularConfig(
            task=task, n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            class_balance=class_balance, noise=noise, seed=42,
            quirks=DEFAULT_QUIRKS if inject else (),
        )
        df = generate(cfg)
        return df, TARGET, {"source": "synthetic", "synth_config": asdict(cfg)}

    def _ask_target(self, ctx: StageContext, df: pd.DataFrame) -> str:
        cols = [str(c) for c in df.columns]
        return ctx.questioner.choice(
            "Which column is the target (what you want to predict)?", cols, allow_other=False
        )

    def _drive(self, ctx: StageContext):
        q = ctx.questioner
        candidates = list_candidates(self.search_roots)
        if candidates:
            answer = q.choice(
                "Which file holds your data? (pick one or type a full path)",
                [str(p) for p in candidates], allow_other=True,
            )
        else:
            answer = q.text(
                "No CSV/Parquet/Excel files found. Enter the full path to your data file"
            )
        path = Path(answer.strip())
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        df = load_table(path)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "drive", "source_path": str(path)}

    def _huggingface(self, ctx: StageContext):
        q = ctx.questioner
        results = []
        for _ in range(3):
            query = q.text(
                "Describe the dataset you want (a few keywords, e.g. 'credit card fraud')"
            )
            results = self.hf_search(query)
            if results:
                break
            ctx.display(f"No datasets found for '{query}'. Try different words.")
        if not results:
            raise RuntimeError("no HuggingFace datasets found after 3 searches")
        labels = [f"{r.id} — {r.downloads:,} downloads — {r.description[:60]}" for r in results]
        pick = q.choice("Which dataset?", labels, allow_other=False)
        chosen = results[labels.index(pick)]
        ctx.display(f"Downloading **{chosen.id}** from the HuggingFace Hub...")
        df = self.hf_load(chosen.id)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "huggingface", "hf_id": chosen.id}

    def _plots(self, ctx: StageContext, df: pd.DataFrame, profile: dict) -> None:
        pdir = ctx.project.plots_dir
        plots.present(plots.feature_histograms(df), pdir, "raw_histograms")
        plots.present(plots.missing_matrix(df), pdir, "raw_missing")
        t = profile.get("target")
        if t and t["kind"] == "categorical":
            plots.present(
                plots.class_balance(t["counts"], t.get("total")), pdir, "raw_class_balance"
            )
        elif t:
            plots.present(plots.target_distribution(df[t["name"]]), pdir, "raw_target_distribution")
        if df.select_dtypes("number").shape[1] >= 2:
            plots.present(plots.correlation_heatmap(df), pdir, "raw_correlation")

    def _narrate(self, ctx: StageContext, spec: dict, profile: dict) -> None:
        prompt = (
            "Project spec:\n" + json.dumps(spec, indent=2)
            + "\n\nData profile:\n" + json.dumps(profile, indent=2)
        )
        try:
            text = ask_text(
                ctx.llm,
                load_prompt("data", audience=audience(ctx.spec().learning_level)),
                prompt,
            )
        except LLMError as exc:
            text = (
                f"(Couldn't reach Claude for a narrative: {exc}) "
                "Data saved. Next: the [[data cleaning]] audit."
            )
        ctx.display(text)
