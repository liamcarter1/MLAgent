"""Clean stage: audit the raw data, agree the fixes, write clean.py, then read its output."""

from __future__ import annotations

import json

import pandas as pd
from pandas.api import types as ptypes

from mlagent.audit import Issue, audit_tabular
from mlagent.cleaning import describe_step, render_clean_py
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_dataframe
from mlagent.prompts_io import audience, load_prompt
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.stages.data import META_FILE, RAW_FILE

CLEAN_FILE = "data.csv"
CLEAN_REL_PATH = "data/clean/data.csv"
AUDIT_FILE = "audit.json"
PROFILE_CLEAN_FILE = "profile_clean.json"
CLEAN_PY = "clean.py"
MIN_TEST_FRACTION = 0.05
# Frozen independently of config.json's tunable `seed` so the held-out test split is the
# same for every run of this project.
SPLIT_SEED = 42


class CleanStage(ScriptStageBase):
    name = "clean"

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE) or {}
        return (
            (ctx.project.data_clean / CLEAN_FILE).exists()
            and ctx.project.exists(AUDIT_FILE)
            and bool(meta.get("splits"))
            and bool(meta.get("feature_columns"))
        )

    def prepare(self, ctx: StageContext) -> Handoff:
        meta = ctx.project.read_json(META_FILE) or {}
        target = meta.get("target")
        if not target:
            raise RuntimeError("data_meta.json has no target; run the data stage first")
        df = pd.read_csv(ctx.project.data_raw / RAW_FILE)
        before = profile_dataframe(df, target)

        ctx.teaching().preamble("clean", {"target": target, "n_rows": int(len(df))})
        issues = audit_tabular(df, target)
        decisions = self._review_issues(ctx, issues, before)
        steps = self._collect_steps(decisions)

        drops = self._ask_drops(ctx, df, target)
        if drops:
            steps.append({"op": "drop_columns", "params": {"columns": drops}})
        splits = self._ask_splits(ctx)

        ctx.project.write_json(
            AUDIT_FILE,
            {"issues": [i.to_dict() for i in issues], "decisions": decisions, "steps": steps},
        )
        (ctx.project.root / CLEAN_PY).write_text(render_clean_py(steps), encoding="utf-8")
        meta.update(
            {"splits": splits, "dropped_columns": drops, "split_seed": SPLIT_SEED}
        )
        ctx.project.write_json(META_FILE, meta)

        listed = "\n".join(f"- {describe_step(s)}" for s in steps) or "- (no changes)"
        ctx.display(
            f"I wrote `clean.py` with {len(steps)} step(s):\n\n{listed}\n\n"
            "Run it in the next cell. It reads `data/raw/data.csv`, writes "
            "`data/clean/data.csv`, and never touches the raw file — so if you change your "
            "mind you can edit `STEPS` in `clean.py` and run it again."
        )
        return Handoff(
            stage=self.name,
            commands=[[CLEAN_PY]],
            outputs=[CLEAN_REL_PATH, PROFILE_CLEAN_FILE],
        )

    def debrief(self, ctx: StageContext) -> None:
        clean_path = ctx.project.data_clean / CLEAN_FILE
        if not clean_path.exists():
            ctx.display(
                "I can't see `data/clean/data.csv` yet. Run the `clean.py` cell, then run this "
                "cell again."
            )
            return
        meta = ctx.project.read_json(META_FILE) or {}
        target = str(meta.get("target"))
        cleaned = pd.read_csv(clean_path)
        feature_columns = [str(c) for c in cleaned.columns if c != target]
        categorical_columns = [
            c for c in feature_columns
            if ptypes.is_object_dtype(cleaned[c])
            or ptypes.is_string_dtype(cleaned[c])
            or ptypes.is_bool_dtype(cleaned[c])
        ]
        meta.update(
            {
                "clean_path": CLEAN_REL_PATH,
                "clean_n_rows": int(len(cleaned)),
                "clean_n_cols": int(cleaned.shape[1]),
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
            }
        )
        if meta.get("task_type") == "tabular_classification" and target in cleaned.columns:
            labels = sorted(str(v) for v in cleaned[target].dropna().unique())
            meta["n_classes"] = len(labels)
            meta["class_labels"] = labels
        ctx.project.write_json(META_FILE, meta)

        profile = ctx.project.read_json(PROFILE_CLEAN_FILE) or {}
        figures = [
            ctx.project.plots_dir / str(name) for name in (profile.get("figures") or [])
        ]
        ctx.display(self._summary(profile, meta))
        payload = {"before": profile.get("before"), "after": profile.get("after"),
                   "steps": profile.get("steps"), "splits": meta.get("splits")}
        note = ctx.teaching().debrief("clean", payload, figures, fallback="")
        if note:
            ctx.display(note)

    def _collect_steps(self, decisions: list[dict]) -> list[dict]:
        """Approved fixes become steps, unless the fix's column was itself dropped by
        another approved fix; those are recorded as skipped so decisions stay meaningful."""
        dropped_cols: set[str] = set()
        for d in decisions:
            if d["approved"] and d["fix"] and d["fix"]["op"] == "drop_columns":
                dropped_cols.update(d["fix"]["params"].get("columns", []))
        steps: list[dict] = []
        for d in decisions:
            if not d["approved"] or not d["fix"]:
                d["applied"] = False
                continue
            fix = d["fix"]
            col = fix.get("params", {}).get("column")
            if fix["op"] != "drop_columns" and col is not None and col in dropped_cols:
                d["applied"] = False
                d["reason"] = "column dropped"
                continue
            d["applied"] = True
            steps.append(fix)
        return steps

    def _review_issues(self, ctx: StageContext, issues: list[Issue], profile: dict) -> list[dict]:
        if not issues:
            ctx.display("The audit found no problems. Nice and clean.")
            return []
        self._explain(ctx, issues, profile)
        decisions: list[dict] = []
        for issue in issues:
            where = f" in `{issue.column}`" if issue.column else ""
            line = f"**[{issue.severity}] {issue.kind}**{where}: {issue.message}"
            if issue.fix:
                ctx.display(f"{line}\n\nProposed fix: {describe_step(issue.fix)}")
                approved = ctx.questioner.confirm(
                    "Apply this fix?", default=issue.severity != "low"
                )
            else:
                ctx.display(f"{line}\n\nNo automatic fix; noted for the modelling stage.")
                approved = False
            decisions.append(
                {"kind": issue.kind, "column": issue.column, "fix": issue.fix, "approved": approved}
            )
        return decisions

    def _explain(self, ctx: StageContext, issues: list[Issue], profile: dict) -> None:
        payload = json.dumps(
            {
                "issues": [i.to_dict() for i in issues],
                "spec": ctx.spec().to_dict(),
                "profile_summary": {
                    "n_rows": profile["n_rows"],
                    "n_cols": profile["n_cols"],
                    "target": profile.get("target"),
                },
            },
            indent=2,
            default=str,
        )
        try:
            text = ask_text(
                ctx.llm, load_prompt("clean", audience=audience(ctx.learning_level())), payload
            )
        except LLMError as exc:
            text = f"(Couldn't reach Claude for the report card: {exc}) Here are the issues found:"
        ctx.display(text)

    def _ask_drops(self, ctx: StageContext, df: pd.DataFrame, target: str) -> list[str]:
        cols = [str(c) for c in df.columns if c != target]
        raw = ctx.questioner.text(
            "Any other columns to drop before training? (comma-separated names, or leave blank)",
            default="", key="clean.drop_columns",
        )
        chosen = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in chosen if c not in cols]
        if unknown:
            ctx.display(f"Ignoring unknown columns: {', '.join(unknown)}")
        return [c for c in chosen if c in cols]

    def _ask_splits(self, ctx: StageContext) -> dict:
        q = ctx.questioner
        train = q.number("Fraction of rows for training?", default=0.7, minimum=0.5, maximum=0.9,
                         key="clean.train_fraction")
        val = q.number(
            "Fraction for validation (used during tuning)?", default=0.15, minimum=0.05,
            maximum=0.3, key="clean.val_fraction",
        )
        test = round(1 - train - val, 4)
        if test < MIN_TEST_FRACTION:
            val = round(max(MIN_TEST_FRACTION, 1 - MIN_TEST_FRACTION - train), 4)
            test = round(1 - train - val, 4)
            ctx.display(
                f"Adjusted so the [[test set]] keeps at least {MIN_TEST_FRACTION:.0%}: "
                f"train {train:g}, validation {val:g}, test {test:g}."
            )
        return {"train": round(train, 4), "val": round(val, 4), "test": test}

    def _summary(self, profile: dict, meta: dict) -> str:
        before = profile.get("before") or {}
        after = profile.get("after") or {}
        steps = profile.get("steps") or []
        lines = [
            "### Cleaning summary",
            "",
            f"- Rows: {before.get('n_rows', '?')} -> {after.get('n_rows', '?')}",
            f"- Columns: {before.get('n_cols', '?')} -> {after.get('n_cols', '?')}",
            f"- Steps applied: {len(steps)}",
        ]
        lines.extend(f"  - {describe_step(s)}" for s in steps)
        splits = meta.get("splits") or {}
        lines += [
            "",
            f"Splits frozen at train {splits.get('train')}, validation {splits.get('val')}, "
            f"test {splits.get('test')} with seed {meta.get('split_seed', SPLIT_SEED)}, so every "
            "run sees the same rows. Next: choosing a model and generating the "
            "[[training pipeline]].",
        ]
        return "\n".join(lines)
