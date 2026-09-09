"""Data stage: obtain raw tabular data, then hand the user profile.py to run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlagent.datasources.drive import list_candidates, load_table
from mlagent.datasources.hf import load_tabular, search_datasets
from mlagent.profile import profile_markdown
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.synth.tabular import TARGET, SynthTabularConfig, generate
from mlagent.templates_io import COMMON_FILES, copy_common

RAW_FILE = "data.csv"
META_FILE = "data_meta.json"
PROFILE_RAW_FILE = "profile_raw.json"
DEFAULT_QUIRKS = ("missing", "duplicates", "id_column", "categorical", "whitespace", "outliers")
DEFAULT_SEARCH_ROOTS = (Path("/content/drive/MyDrive"), Path("/content"))
TABULAR_TASKS = {"tabular_classification": "classification", "tabular_regression": "regression"}


class DataStage(ScriptStageBase):
    name = "data"

    def __init__(self, search_roots=None, hf_search=search_datasets, hf_load=load_tabular):
        self.search_roots = (
            list(search_roots) if search_roots is not None else list(DEFAULT_SEARCH_ROOTS)
        )
        self.hf_search = hf_search
        self.hf_load = hf_load

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE)
        return (
            bool(meta and meta.get("target"))
            and (ctx.project.data_raw / RAW_FILE).exists()
            and ctx.project.exists(PROFILE_RAW_FILE)
        )

    def prepare(self, ctx: StageContext) -> Handoff:
        spec = ctx.spec()
        if spec.task_type not in TABULAR_TASKS:
            raise NotImplementedError(
                f"{spec.task_type} data is not supported yet (image tasks arrive in Milestone 6)"
            )
        ctx.teaching().preamble("data", {"spec": spec.to_dict()})
        if spec.data_source == "synthetic":
            df, target, meta = self._synthetic(ctx, TABULAR_TASKS[spec.task_type])
        elif spec.data_source == "drive":
            df, target, meta = self._drive(ctx)
        else:
            df, target, meta = self._huggingface(ctx)

        ctx.project.data_raw.mkdir(parents=True, exist_ok=True)
        path = ctx.project.data_raw / RAW_FILE
        df.to_csv(path, index=False)
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
        copy_common(COMMON_FILES, ctx.project.root)
        ctx.display(
            f"I saved {len(df)} rows and {df.shape[1]} columns to `data/raw/data.csv` and wrote "
            "`profile.py`, which measures the data and draws four figures. Run it in the next "
            "cell; nothing about the raw data is changed."
        )
        return Handoff(
            stage=self.name, commands=[["profile.py"]], outputs=[PROFILE_RAW_FILE]
        )

    def debrief(self, ctx: StageContext) -> None:
        profile = ctx.project.read_json(PROFILE_RAW_FILE)
        if not isinstance(profile, dict):
            ctx.display(
                "I can't see `profile_raw.json` yet. Run the `profile.py` cell, then run this "
                "cell again."
            )
            return
        ctx.display(profile_markdown(profile))
        figures = [
            ctx.project.plots_dir / str(name) for name in (profile.get("figures") or [])
        ]
        payload = {
            "spec": ctx.spec().to_dict(),
            "profile": {k: v for k, v in profile.items() if k != "figures"},
        }
        fallback = "Data saved. Next: the [[data cleaning]] audit."
        ctx.display(ctx.teaching().debrief("data", payload, figures, fallback=fallback))

    def _synthetic(self, ctx: StageContext, task: str):
        q = ctx.questioner
        n_samples = int(q.number("How many rows?", default=1000, minimum=100, maximum=200000,
                                 key="data.n_rows"))
        n_features = int(q.number("How many numeric features?", default=8, minimum=2,
                                  maximum=100, key="data.n_features"))
        n_classes, class_balance = 2, 0.5
        if task == "classification":
            n_classes = int(q.number("How many classes?", default=2, minimum=2, maximum=10,
                                     key="data.n_classes"))
            if n_classes == 2:
                class_balance = q.number(
                    "Fraction of rows in the majority class (0.5 = balanced)?",
                    default=0.5, minimum=0.5, maximum=0.95, key="data.class_balance",
                )
        noise = q.number(
            "Label/measurement noise (0 = clean, 0.3 = very noisy)?",
            default=0.1, minimum=0.0, maximum=1.0, key="data.noise",
        )
        inject = q.confirm(
            "Inject realistic data problems (missing values, duplicates, an ID column, messy "
            "categories, outliers) so the cleaning stage has work to do?",
            default=True, key="data.inject_quirks",
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
            "Which column is the target (what you want to predict)?", cols, allow_other=False,
            key="data.target_column",
        )

    def _drive(self, ctx: StageContext):
        q = ctx.questioner
        candidates = list_candidates(self.search_roots)
        if candidates:
            answer = q.choice(
                "Which file holds your data? (pick one or type a full path)",
                [str(p) for p in candidates], allow_other=True, key="data.drive_path",
            )
        else:
            answer = q.text(
                "No CSV/Parquet/Excel files found. Enter the full path to your data file",
                key="data.drive_path",
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
                "Describe the dataset you want (a few keywords, e.g. 'credit card fraud')",
                key="data.hf_query",
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
