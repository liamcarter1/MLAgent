"""Clean stage: audit the raw data, explain the issues, apply approved fixes, record splits."""

from __future__ import annotations

import json

import pandas as pd

from mlagent import plots
from mlagent.audit import Issue, audit_tabular
from mlagent.cleaning import apply_steps, describe_step, render_clean_py
from mlagent.llm import LLMError, ask_text
from mlagent.profile import profile_dataframe
from mlagent.prompts_io import load_prompt
from mlagent.stages.base import StageContext
from mlagent.stages.data import META_FILE, RAW_FILE

CLEAN_FILE = "data.csv"
AUDIT_FILE = "audit.json"
PROFILE_CLEAN_FILE = "profile_clean.json"
CLEAN_PY = "clean.py"
MIN_TEST_FRACTION = 0.05


class CleanStage:
    name = "clean"

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE) or {}
        return (
            (ctx.project.data_clean / CLEAN_FILE).exists()
            and ctx.project.exists(AUDIT_FILE)
            and bool(meta.get("splits"))
        )

    def run(self, ctx: StageContext) -> None:
        meta = ctx.project.read_json(META_FILE) or {}
        target = meta.get("target")
        if not target:
            raise RuntimeError("data_meta.json has no target; run the data stage first")
        df = pd.read_csv(ctx.project.data_raw / RAW_FILE)
        before = profile_dataframe(df, target)

        issues = audit_tabular(df, target)
        decisions = self._review_issues(ctx, issues, before)
        steps = [d["fix"] for d in decisions if d["approved"] and d["fix"]]
        cleaned = apply_steps(df, steps)

        drops = self._ask_drops(ctx, cleaned, target)
        if drops:
            step = {"op": "drop_columns", "params": {"columns": drops}}
            steps.append(step)
            cleaned = apply_steps(cleaned, [step])
        splits = self._ask_splits(ctx)

        ctx.project.data_clean.mkdir(parents=True, exist_ok=True)
        clean_path = ctx.project.data_clean / CLEAN_FILE
        cleaned.to_csv(clean_path, index=False)
        (ctx.project.root / CLEAN_PY).write_text(render_clean_py(steps), encoding="utf-8")
        after = profile_dataframe(cleaned, target)
        ctx.project.write_json(PROFILE_CLEAN_FILE, after)
        ctx.project.write_json(
            AUDIT_FILE,
            {
                "issues": [i.to_dict() for i in issues],
                "decisions": decisions,
                "steps": steps,
            },
        )
        meta.update({"splits": splits, "clean_path": str(clean_path), "dropped_columns": drops})
        ctx.project.write_json(META_FILE, meta)

        plots.present(
            plots.before_after_missing(before, after),
            ctx.project.plots_dir,
            "clean_before_after_missing",
        )
        ctx.display(self._summary(before, after, steps))

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
            text = ask_text(ctx.llm, load_prompt("clean"), payload)
        except LLMError as exc:
            text = f"(Couldn't reach Claude for the report card: {exc}) Here are the issues found:"
        ctx.display(text)

    def _ask_drops(self, ctx: StageContext, df: pd.DataFrame, target: str) -> list[str]:
        cols = [str(c) for c in df.columns if c != target]
        raw = ctx.questioner.text(
            "Any other columns to drop before training? (comma-separated names, or leave blank)",
            default="",
        )
        chosen = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in chosen if c not in cols]
        if unknown:
            ctx.display(f"Ignoring unknown columns: {', '.join(unknown)}")
        return [c for c in chosen if c in cols]

    def _ask_splits(self, ctx: StageContext) -> dict:
        q = ctx.questioner
        train = q.number("Fraction of rows for training?", default=0.7, minimum=0.5, maximum=0.9)
        val = q.number(
            "Fraction for validation (used during tuning)?", default=0.15, minimum=0.05, maximum=0.3
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

    def _summary(self, before: dict, after: dict, steps: list[dict]) -> str:
        lines = [
            "### Cleaning summary",
            "",
            f"- Rows: {before['n_rows']} → {after['n_rows']}",
            f"- Columns: {before['n_cols']} → {after['n_cols']}",
            f"- Steps applied: {len(steps)}",
        ]
        lines.extend(f"  - {describe_step(s)}" for s in steps)
        lines += [
            "",
            "Cleaned data saved to `data/clean/data.csv`; the steps are in `clean.py` so they can "
            "be re-run on new data. Next: generating the [[training pipeline]].",
        ]
        return "\n".join(lines)
