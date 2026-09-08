"""Locate, copy and validate the reference training templates."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
CODE_FILES = ("data.py", "model.py", "train.py")
SCHEMA_FILE = "config_schema.json"
TEMPLATE_FOR_TASK = {
    "tabular_classification": "tabular_sklearn",
    "tabular_regression": "tabular_sklearn",
}

COMMON_DIRNAME = "common"
COMMON_DIR = TEMPLATES_DIR / COMMON_DIRNAME
# Scripts copied verbatim into a project via copy_common(). `clean.py` also lives under
# COMMON_DIR but is never copied as-is: cleaning.render_clean_py() reads it and
# substitutes the approved steps into its STEPS_JSON line before writing it out.
COMMON_FILES = ("profile.py",)


def template_dir(name: str) -> Path:
    path = TEMPLATES_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(f"no template named {name!r} under {TEMPLATES_DIR}")
    return path


def load_schema(name: str) -> dict:
    return json.loads((template_dir(name) / SCHEMA_FILE).read_text(encoding="utf-8"))


def model_types(schema: dict) -> list[str]:
    return list((schema.get("models") or {}).keys())


def schema_for(schema: dict, model_type: str) -> dict:
    """The flat rules for one model family: the common keys plus that family's keys.

    Everything downstream — `default_config`, `validate_config`, `coerce_config`, the
    codegen proposal and Milestone 5's tuner — works on this flat form.
    """
    models = schema.get("models") or {}
    if model_type not in models:
        raise ValueError(
            f"unknown model_type {model_type!r}; expected one of {sorted(models)}"
        )
    return {**(schema.get("common") or {}), **models[model_type]}


def default_config(schema: dict) -> dict:
    return {key: rule.get("default") for key, rule in schema.items()}


def _check_value(key: str, value, rule: dict) -> str | None:
    if value is None:
        return None if rule.get("nullable") else f"{key}: must not be null"
    if rule.get("type") == "choice":
        choices = list(rule.get("choices") or [])
        if not isinstance(value, str):
            return f"{key}: expected one of {choices}, got {value!r}"
        if value not in choices:
            return f"{key}: {value!r} is not one of {choices}"
        return None
    if isinstance(value, bool):
        return f"{key}: expected a number, got a boolean"
    if rule.get("type") == "integer":
        if not isinstance(value, int) and not (isinstance(value, float) and value.is_integer()):
            return f"{key}: expected an integer, got {value!r}"
    elif not isinstance(value, int | float):
        return f"{key}: expected a number, got {value!r}"
    if isinstance(value, float) and not math.isfinite(value):
        return f"{key}: {value!r} is not a finite number"
    if rule.get("min") is not None and value < rule["min"]:
        return f"{key}: {value!r} is below the minimum {rule['min']}"
    if rule.get("max") is not None and value > rule["max"]:
        return f"{key}: {value!r} is above the maximum {rule['max']}"
    return None


def validate_config(config: dict, schema: dict) -> list[str]:
    """Return a list of problems; empty means the config is valid against the schema."""
    problems: list[str] = []
    for key in config:
        if key not in schema:
            problems.append(f"{key}: not a tunable key")
    for key, rule in schema.items():
        if key not in config:
            problems.append(f"{key}: missing")
            continue
        problem = _check_value(key, config[key], rule)
        if problem:
            problems.append(problem)
    return problems


def _cast(value, rule: dict):
    if isinstance(value, bool):
        raise TypeError("boolean")
    if value is None:
        return None
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("not finite")
    return int(round(value)) if rule.get("type") == "integer" else float(value)


def coerce_config(proposal: dict, schema: dict) -> tuple[dict, list[str]]:
    """Overlay a proposal on the defaults, clamping out-of-range values and dropping
    unknown keys. Notes describe every adjustment in plain words."""
    if proposal is not None and not isinstance(proposal, dict):
        return default_config(schema), ["Ignored proposal: expected an object of key -> value."]
    config = default_config(schema)
    notes: list[str] = []
    for key, value in (proposal or {}).items():
        if key not in schema:
            notes.append(f"Ignored unknown key {key!r}.")
            continue
        rule = schema[key]
        if rule.get("type") == "choice":
            choices = list(rule.get("choices") or [])
            if isinstance(value, str) and value in choices:
                config[key] = value
            else:
                notes.append(
                    f"Ignored {key}={value!r}: not one of {choices}; kept {config[key]!r}."
                )
            continue
        try:
            cast = _cast(value, rule)
        except (TypeError, ValueError):
            notes.append(f"Ignored {key}={value!r}: not a number; kept {config[key]!r}.")
            continue
        if cast is None:
            if rule.get("nullable"):
                config[key] = None
            else:
                notes.append(f"Ignored null for {key}; kept {config[key]!r}.")
            continue
        low, high = rule.get("min"), rule.get("max")
        if low is not None and cast < low:
            notes.append(f"Raised {key} from {cast!r} to the minimum {low!r}.")
            cast = _cast(low, rule)
        elif high is not None and cast > high:
            notes.append(f"Lowered {key} from {cast!r} to the maximum {high!r}.")
            cast = _cast(high, rule)
        config[key] = cast
    return config, notes


def copy_template(name: str, project_root: Path) -> list[Path]:
    """Copy the template's code files into the project folder, overwriting; return the paths."""
    src = template_dir(name)
    written: list[Path] = []
    for filename in CODE_FILES:
        target = Path(project_root) / filename
        shutil.copyfile(src / filename, target)
        written.append(target)
    return written


def common_file(name: str) -> Path:
    """Path to a shared template script (`profile.py`, `clean.py`)."""
    path = COMMON_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"no common template named {name!r} under {COMMON_DIR}")
    return path


def copy_common(names: Sequence[str], project_root: Path) -> list[Path]:
    """Copy shared scripts into the project folder, overwriting; return the paths."""
    written: list[Path] = []
    for filename in names:
        target = Path(project_root) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common_file(filename), target)
        written.append(target)
    return written
