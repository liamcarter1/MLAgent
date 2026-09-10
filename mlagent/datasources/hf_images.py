"""Load an image classification dataset from the HuggingFace Hub.

The `datasets` import is lazy (as `datasources/hf.py` does for tables) so importing
`mlagent` never requires it installed, and `loader` is injectable so tests never hit the
network.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from mlagent.imageset import ImageSet, make_manifest, prepare_image, stratified_indices

IMAGE_COLUMN_NAMES = ("image", "img", "picture")
LABEL_COLUMN_NAMES = ("label", "labels", "fine_label", "class")


def _detect_image_column(features: dict, given: str | None) -> str:
    if given:
        return given
    for name, feature in features.items():
        if type(feature).__name__ == "Image":
            return name
    for name in IMAGE_COLUMN_NAMES:
        if name in features:
            return name
    raise ValueError(
        f"could not find an image column in {sorted(features)}; pass image_column="
    )


def _detect_label_column(features: dict, given: str | None) -> str:
    if given:
        return given
    for name, feature in features.items():
        if hasattr(feature, "names"):
            return name
    for name in LABEL_COLUMN_NAMES:
        if name in features:
            return name
    raise ValueError(
        f"could not find a label column in {sorted(features)}; pass label_column="
    )


def _is_int_label(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _class_names_and_index(
    features: dict, label_column: str, raw_labels: list
) -> tuple[list[str], dict]:
    """Class names plus a `{canonical raw value: position}` index built from the same
    values, so every row's label is always looked up rather than reused directly.

    A `ClassLabel`-style `.names` list is trusted as-is: a raw int value is its position in
    `names` (by HuggingFace convention) and a raw string value is looked up by name, so
    either is accepted. Otherwise, if every raw value is an int, class names are the
    numerically sorted distinct values (so 10 classes sort as 0..9, not lexicographically);
    otherwise class names are the sorted distinct strings.
    """
    feature = features.get(label_column)
    names = getattr(feature, "names", None)
    if names:
        class_names = [str(n) for n in names]
        index: dict = {}
        for i, name in enumerate(class_names):
            index[name] = i
            index[i] = i
        return class_names, index
    if raw_labels and all(_is_int_label(v) for v in raw_labels):
        distinct = sorted({int(v) for v in raw_labels})
        class_names = [str(v) for v in distinct]
        return class_names, {v: i for i, v in enumerate(distinct)}
    distinct = sorted({str(v) for v in raw_labels})
    return distinct, {v: i for i, v in enumerate(distinct)}


def _to_pil(value, row: int, column: str):
    """Decode one dataset row's image value to a loaded `PIL.Image`.

    Handles an already-decoded `PIL.Image`, a HuggingFace `{"bytes": ...}` or
    `{"path": ...}` dict, raw bytes, and a file path string. Anything else, or a decode
    failure, raises a `ValueError` naming the row and column so the user knows which
    dataset column to override with `image_column=`.
    """
    from PIL import Image as PILImage

    try:
        if isinstance(value, PILImage.Image):
            img = value
        elif isinstance(value, dict) and value.get("bytes") is not None:
            img = PILImage.open(io.BytesIO(value["bytes"]))
        elif isinstance(value, dict) and value.get("path") is not None:
            img = PILImage.open(value["path"])
        elif isinstance(value, (bytes, bytearray)):
            img = PILImage.open(io.BytesIO(value))
        elif isinstance(value, (str, Path)):
            img = PILImage.open(value)
        else:
            raise ValueError(
                f"row {row}, column {column!r}: cannot decode a {type(value).__name__} "
                "as an image; pass image_column= explicitly"
            )
        img.load()
        return img
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"row {row}, column {column!r}: could not decode a "
            f"{type(value).__name__} as an image ({exc}); pass image_column= explicitly"
        ) from exc


def load_image_dataset(
    dataset_id: str,
    image_size: int,
    split: str = "train",
    max_images: int | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
    loader=None,
) -> ImageSet:
    """One split of a HuggingFace image dataset, normalised to an ImageSet."""
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    ds = loader(dataset_id, split=split)
    if hasattr(ds, "keys") and not hasattr(ds, "features"):
        key = split if split in ds else next(iter(ds.keys()))
        ds = ds[key]

    features = dict(getattr(ds, "features", {}) or {})
    image_key = _detect_image_column(features, image_column)
    label_key = _detect_label_column(features, label_column)

    rows = [ds[i] for i in range(len(ds))]
    raw_labels = [row[label_key] for row in rows]
    class_names, index = _class_names_and_index(features, label_key, raw_labels)

    arrays: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    sizes: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        image = _to_pil(row[image_key], i, image_key)
        width, height = image.size
        arrays.append(prepare_image(image, image_size))
        value = raw_labels[i]
        key = int(value) if _is_int_label(value) else str(value)
        labels.append(index[key])
        sources.append(f"hf:{dataset_id}/{split}#{i}")
        sizes.append((int(width), int(height)))

    if not arrays:
        raise ValueError(f"{dataset_id} split {split!r} has no rows")

    label_array = np.array(labels, dtype=np.int64)
    imageset = ImageSet(
        images=np.stack(arrays),
        labels=label_array,
        class_names=class_names,
        manifest=make_manifest(label_array, class_names, sources, sizes),
    )
    keep = stratified_indices(imageset.labels, max_images)
    if len(keep) != imageset.n_images:
        imageset = imageset.take(keep)
    imageset.validate()
    return imageset
