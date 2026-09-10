"""Load an image classification dataset from the HuggingFace Hub.

The `datasets` import is lazy (as `datasources/hf.py` does for tables) so importing
`mlagent` never requires it installed, and `loader` is injectable so tests never hit the
network.
"""

from __future__ import annotations

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


def _class_names(features: dict, label_column: str, raw_labels: list) -> list[str]:
    feature = features.get(label_column)
    names = getattr(feature, "names", None)
    if names:
        return [str(n) for n in names]
    return sorted({str(v) for v in raw_labels})


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
    class_names = _class_names(features, label_key, raw_labels)
    index = {name: i for i, name in enumerate(class_names)}

    arrays: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    sizes: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        image = row[image_key]
        width, height = getattr(image, "size", (image_size, image_size))
        arrays.append(prepare_image(image, image_size))
        value = raw_labels[i]
        labels.append(int(value) if isinstance(value, int) else index[str(value)])
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
