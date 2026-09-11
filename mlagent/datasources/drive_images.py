"""Load a folder of images laid out as `<class>/<file>` (Drive, or any directory).

Nothing here ever raises on a bad file: unreadable files, non-image files and images
dropped straight into the root with no class folder are collected in `skipped` as
`(path, reason)` pairs so the audit can report them to the user in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mlagent.imageset import ImageSet, make_manifest, prepare_image, stratified_indices

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}
ROOT_LEVEL_REASON = "no class folder"


def _candidate_files(root: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """(class_name, path) pairs plus the root-level strays, both sorted by path."""
    found: list[tuple[str, Path]] = []
    skipped: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        here = Path(dirpath)
        for name in sorted(filenames):
            path = here / name
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:  # pragma: no cover - os.walk always stays under root
                continue
            if len(relative.parts) < 2:
                skipped.append((str(path), ROOT_LEVEL_REASON))
                continue
            found.append((relative.parts[0], path))
    return found, skipped


def load_folder(
    path: Path | str, image_size: int, max_images: int | None = None
) -> tuple[ImageSet, list[tuple[str, str]]]:
    """Read `<class>/<file>` images into an ImageSet, plus the files that could not be read."""
    from PIL import Image

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"no such folder: {root}")
    candidates, skipped = _candidate_files(root)

    # Cap on the class folder names alone, before opening a single file, so a `max_images`
    # cap skips reading (and decoding) the files that would be dropped anyway.
    class_names_all = [name for name, _p in candidates]
    distinct = sorted(set(class_names_all))
    class_index = {name: i for i, name in enumerate(distinct)}
    labels_all = np.array([class_index[name] for name in class_names_all], dtype=np.int64)
    keep = stratified_indices(labels_all, max_images)
    candidates = [candidates[i] for i in keep]

    arrays: list[np.ndarray] = []
    class_of: list[str] = []
    sources: list[str] = []
    sizes: list[tuple[int, int]] = []
    for class_name, file_path in candidates:
        try:
            with Image.open(file_path) as img:
                img.load()
                width, height = img.size
                arrays.append(prepare_image(img, image_size))
        except Exception as exc:  # noqa: BLE001 - any unreadable file is reported, never fatal
            skipped.append((str(file_path), f"{type(exc).__name__}: {exc}"))
            continue
        class_of.append(class_name)
        sources.append(str(file_path))
        sizes.append((width, height))

    if not arrays:
        raise ValueError(f"no readable images under {root} (looked for {list(IMAGE_EXTS)})")

    class_names = sorted(set(class_of))
    index = {name: i for i, name in enumerate(class_names)}
    labels = np.array([index[name] for name in class_of], dtype=np.int64)
    imageset = ImageSet(
        images=np.stack(arrays),
        labels=labels,
        class_names=class_names,
        manifest=make_manifest(labels, class_names, sources, sizes),
    )
    imageset.validate()
    return imageset, skipped
