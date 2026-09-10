"""Cleanliness audit for image datasets.

The image counterpart of `audit.py`: each check returns `audit.Issue`s with evidence and,
where a fix is possible, a proposed step. Dropping images is the only automatic fix an
image dataset supports, so every `fix` here is a `drop_indices` step.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlagent.audit import SEVERITY_ORDER, Issue
from mlagent.imageset import ImageSet, blank_indices, duplicate_groups

# Mirrors audit.py's tabular class-balance threshold.
MAJORITY_LIMIT = 0.9
MIN_IMAGES_PER_CLASS = 10
# A source image whose shorter side is below image_size * this was upscaled to fit.
UPSCALE_RATIO = 0.5
MAX_LISTED = 20


def _drop(indices: list[int], reason: str) -> dict:
    return {"op": "drop_indices", "params": {"indices": [int(i) for i in indices],
                                             "reason": reason}}


def _check_balance(imageset: ImageSet, _meta: dict) -> list[Issue]:
    counts = imageset.class_counts()
    total = sum(counts.values())
    if not total:
        return []
    out: list[Issue] = []
    majority = max(counts.values()) / total
    if majority > MAJORITY_LIMIT:
        out.append(Issue(
            "class_imbalance", "medium",
            f"The biggest class holds {majority:.0%} of the images; accuracy will look "
            "good even for a model that always guesses that class.",
            None,
            {"majority_fraction": round(float(majority), 4), "counts": dict(counts)},
        ))
    tiny = {name: n for name, n in counts.items() if n < MIN_IMAGES_PER_CLASS}
    if tiny:
        out.append(Issue(
            "tiny_classes", "high",
            f"{len(tiny)} class(es) have fewer than {MIN_IMAGES_PER_CLASS} images "
            f"({', '.join(f'{k}: {v}' for k, v in tiny.items())}). There is no automatic "
            "fix: collect more examples of these classes, or drop them from the dataset.",
            None,
            {"classes": tiny, "minimum": MIN_IMAGES_PER_CLASS},
        ))
    return out


def _check_duplicates(imageset: ImageSet, _meta: dict) -> list[Issue]:
    groups = duplicate_groups(imageset.images)
    if not groups:
        return []
    extras = sorted(i for group in groups for i in group[1:])
    return [Issue(
        "duplicate_images", "medium",
        f"{len(extras)} image(s) are an exact copy of another image. Duplicates that land "
        "in different splits leak the answer from training into validation.",
        None,
        {"groups": len(groups), "extra_copies": len(extras),
         "example_group": groups[0][:MAX_LISTED]},
        _drop(extras, "duplicate images"),
    )]


def _check_blanks(imageset: ImageSet, _meta: dict) -> list[Issue]:
    blanks = blank_indices(imageset.images)
    if not blanks:
        return []
    return [Issue(
        "blank_images", "medium",
        f"{len(blanks)} image(s) are a near-constant block of colour with nothing in them. "
        "They teach the model nothing and dilute the class they sit in.",
        None,
        {"count": len(blanks), "examples": blanks[:MAX_LISTED]},
        _drop(blanks, "blank images"),
    )]


def _check_upscaled(imageset: ImageSet, meta: dict) -> list[Issue]:
    size = int(meta.get("image_size") or imageset.image_size or 0)
    manifest = imageset.manifest
    if not size or "source_width" not in manifest or "source_height" not in manifest:
        return []
    shorter = np.minimum(
        manifest["source_width"].to_numpy(dtype=float),
        manifest["source_height"].to_numpy(dtype=float),
    )
    small = np.flatnonzero(shorter < size * UPSCALE_RATIO)
    if small.size == 0:
        return []
    return [Issue(
        "upscaled_sources", "low",
        f"{small.size} source image(s) were smaller than {int(size * UPSCALE_RATIO)}px on "
        f"their shorter side and had to be blown up to {size}px, so they carry less detail "
        "than the rest. Nothing to fix; just do not expect much from them.",
        None,
        {"count": int(small.size), "image_size": size,
         "examples": [int(i) for i in small[:MAX_LISTED].tolist()]},
    )]


def _check_skipped(skipped: Sequence[tuple[str, str]]) -> list[Issue]:
    items = list(skipped or ())
    if not items:
        return []
    return [Issue(
        "unreadable_files", "low",
        f"{len(items)} file(s) in the source folder were skipped: they were not readable "
        "images, or sat outside a class subfolder. They are not part of the dataset.",
        None,
        {"count": len(items),
         "files": [{"path": str(p), "reason": str(r)} for p, r in items[:MAX_LISTED]]},
    )]


CHECKS = (_check_balance, _check_duplicates, _check_blanks, _check_upscaled)


def audit_images(
    imageset: ImageSet, meta: dict, skipped: Sequence[tuple[str, str]] = ()
) -> list[Issue]:
    """Every problem worth telling the user about, highest severity first."""
    issues: list[Issue] = []
    for check in CHECKS:
        issues.extend(check(imageset, meta or {}))
    issues.extend(_check_skipped(skipped))
    issues.sort(key=lambda i: SEVERITY_ORDER[i.severity])
    return issues
