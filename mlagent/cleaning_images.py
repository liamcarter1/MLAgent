"""Image cleaning steps as data.

`apply_steps()` runs them; `render_clean_py()` writes a re-runnable script that calls the
same operation. The mirror of `cleaning.py`, but the only operation an image dataset
supports is dropping images by index.
"""

from __future__ import annotations

import json

from mlagent.imageset import ImageSet
from mlagent.templates_io import IMAGE_COMMON_DIRNAME, shared_file

CLEAN_TEMPLATE = f"{IMAGE_COMMON_DIRNAME}/clean.py"
STEPS_MARKER = 'STEPS_JSON = r"""[]"""'
OPS = ("drop_indices",)


def apply_steps(imageset: ImageSet, steps: list[dict]) -> ImageSet:
    """Drop every index any approved step names, once, keeping the original order."""
    if not steps:
        return imageset.take(range(imageset.n_images))
    dropped: set[int] = set()
    for step in steps:
        op = step.get("op")
        if op not in OPS:
            raise ValueError(f"unknown image cleaning op {op!r}")
        dropped.update(int(i) for i in (step.get("params") or {}).get("indices", []))
    keep = [i for i in range(imageset.n_images) if i not in dropped]
    return imageset.take(keep)


def describe_step(step: dict) -> str:
    params = step.get("params", {})
    if step.get("op") == "drop_indices":
        n = len(params.get("indices", []))
        reason = params.get("reason") or "flagged by the audit"
        return f"drop {n} image(s) ({reason})"
    return f"{step.get('op')} {params}"


def render_clean_py(steps: list[dict]) -> str:
    """The standalone image-cleaning script for this project: the shared template with the
    approved steps substituted into its one placeholder line."""
    source = shared_file(CLEAN_TEMPLATE).read_text(encoding="utf-8")
    if STEPS_MARKER not in source:
        raise ValueError(
            f"templates/{CLEAN_TEMPLATE} no longer contains the line {STEPS_MARKER!r}"
        )
    payload = json.dumps(steps, indent=2)
    return source.replace(STEPS_MARKER, f'STEPS_JSON = r"""{payload}"""', 1)
