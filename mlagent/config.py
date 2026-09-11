"""Project-wide constants. Override the model with the MLAGENT_MODEL env var."""

import os

MODEL_ID: str = os.environ.get("MLAGENT_MODEL", "claude-opus-5")
MAX_TOKENS: int = 16000
EFFORT: str = "medium"  # low | medium | high | xhigh | max
MAX_TOOL_ROUNDS: int = 20

DRIVE_ROOT: str = "/content/drive/MyDrive/ml_agent"
PROJECTS_DIRNAME: str = "projects"

SPEC_FILE = "spec.json"
STATE_FILE = "state.json"
GLOSSARY_FILE = "glossary.json"
CONFIG_FILE = "config.json"
RUNS_FILE = "runs.jsonl"
METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"
REPORT_META_FILE = "report_meta.json"
RUNS_DIRNAME = "runs"
TUNE_STATE_FILE = "tune_state.json"

COST_FILE = "cost.json"
DRY_RUN_FILE = "dry_run.json"
