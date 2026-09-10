"""Images in memory and on disk.

The tabular side of the pipeline passes a `pandas.DataFrame` and stores `data.csv`; the
image side passes an `ImageSet` and stores `data.npz` plus `manifest.csv` in the same
folder. Every source (synthetic, Drive, HuggingFace) normalises to the same shape: RGB
uint8 `(N, H, W, 3)` with `H == W == image_size`, integer labels indexing a sorted
`class_names` list, and one manifest row per image recording where it came from and how
big it was *before* resizing.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = "data.npz"
MANIFEST_FILE = "manifest.csv"
MANIFEST_COLUMNS = (
    "index", "label", "class_name", "source", "source_width", "source_height",
)
# Per-image pixel standard deviation (0-255 scale) below which an image counts as blank.
BLANK_STD_THRESHOLD = 3.0
IMAGE_SIZES = (32, 64, 128)
DEFAULT_IMAGE_SIZE = 64


@dataclass
class ImageSet:
    images: np.ndarray       # uint8, shape (N, H, W, 3)
    labels: np.ndarray       # int, shape (N,)
    class_names: list[str]   # sorted; labels index into this list
    manifest: pd.DataFrame

    @property
    def n_images(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.images.shape[1]) if self.images.ndim == 4 else 0

    @property
    def n_channels(self) -> int:
        return int(self.images.shape[3]) if self.images.ndim == 4 else 0

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for label in np.asarray(self.labels).tolist():
            if 0 <= int(label) < len(self.class_names):
                counts[self.class_names[int(label)]] += 1
        return counts

    def validate(self) -> None:
        """Raise ValueError if the arrays and the manifest do not line up."""
        if self.images.ndim != 4 or self.images.shape[3] != 3:
            raise ValueError(f"images must be (N, H, W, 3); got {self.images.shape}")
        if self.images.dtype != np.uint8:
            raise ValueError(f"images must be uint8; got {self.images.dtype}")
        if self.images.shape[1] != self.images.shape[2]:
            raise ValueError("images must be square; run prepare_image on every source")
        if len(self.labels) != self.n_images:
            raise ValueError(f"{len(self.labels)} labels for {self.n_images} images")
        labels_arr = np.asarray(self.labels)
        if not np.issubdtype(labels_arr.dtype, np.integer):
            raise ValueError(f"labels must be an integer dtype; got {labels_arr.dtype}")
        if labels_arr.size and (labels_arr.min() < 0 or labels_arr.max() >= len(self.class_names)):
            raise ValueError(
                f"labels must be in [0, {len(self.class_names)}) to index class_names; "
                f"got min={int(labels_arr.min())}, max={int(labels_arr.max())}"
            )
        if len(self.manifest) != self.n_images:
            raise ValueError(f"{len(self.manifest)} manifest rows for {self.n_images} images")
        if not self.class_names:
            raise ValueError("class_names is empty")

    def take(self, indices) -> ImageSet:
        """A new ImageSet keeping only `indices`, in order, with the manifest reindexed."""
        idx = [int(i) for i in indices]
        manifest = self.manifest.iloc[idx].copy().reset_index(drop=True)
        manifest["index"] = range(len(manifest))
        return ImageSet(
            images=self.images[idx],
            labels=np.asarray(self.labels)[idx],
            class_names=list(self.class_names),
            manifest=manifest,
        )


def make_manifest(labels, class_names: list[str], sources, sizes) -> pd.DataFrame:
    """One row per image: where it came from and its pre-resize width and height."""
    values = [int(v) for v in np.asarray(labels).tolist()]
    pairs = [(int(w), int(h)) for w, h in sizes]
    return pd.DataFrame(
        {
            "index": list(range(len(values))),
            "label": values,
            "class_name": [class_names[v] for v in values],
            "source": list(sources),
            "source_width": [w for w, _h in pairs],
            "source_height": [h for _w, h in pairs],
        },
        columns=list(MANIFEST_COLUMNS),
    )


def write_pair(imageset: ImageSet, directory: Path) -> tuple[Path, Path]:
    """Write `data.npz` (images, labels, class_names) and `manifest.csv` into `directory`.

    `validate()` enforces the uint8/integer dtype contract before anything is written, so
    this never silently casts (and truncates) a mistyped array.
    """
    imageset.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / RAW_FILE
    np.savez_compressed(
        npz_path,
        images=np.ascontiguousarray(imageset.images),
        labels=np.asarray(imageset.labels),
        class_names=np.asarray(imageset.class_names, dtype="U"),
    )
    manifest_path = directory / MANIFEST_FILE
    imageset.manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    return npz_path, manifest_path


def read_pair(directory: Path) -> ImageSet:
    """Read back what `write_pair` wrote."""
    directory = Path(directory)
    with np.load(directory / RAW_FILE) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(v) for v in data["class_names"].tolist()]
    manifest = pd.read_csv(directory / MANIFEST_FILE)
    return ImageSet(images=images, labels=labels, class_names=class_names, manifest=manifest)


def prepare_image(pil_image, image_size: int) -> np.ndarray:
    """RGB, centre-cropped to a square, resized to `image_size`, as uint8 (H, W, 3)."""
    from PIL import Image

    img = pil_image.convert("RGB")
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((int(image_size), int(image_size)), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def image_hash(image: np.ndarray) -> str:
    """A content hash of one image's raw bytes; equal hashes are exact duplicates."""
    return hashlib.sha1(np.ascontiguousarray(image, dtype=np.uint8).tobytes()).hexdigest()


def duplicate_groups(images: np.ndarray) -> list[list[int]]:
    """Groups of indices sharing an exact content hash, largest group first, then by first
    index. Images with no duplicate are not listed."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, image in enumerate(np.asarray(images)):
        buckets[image_hash(image)].append(i)
    groups = [sorted(idx) for idx in buckets.values() if len(idx) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def blank_indices(images: np.ndarray, threshold: float = BLANK_STD_THRESHOLD) -> list[int]:
    """Indices of near-constant images: per-image pixel standard deviation below
    `threshold` on the 0-255 scale."""
    arr = np.asarray(images, dtype=np.float32)
    if arr.size == 0:
        return []
    stds = arr.reshape(arr.shape[0], -1).std(axis=1)
    return [int(i) for i in np.flatnonzero(stds < float(threshold)).tolist()]


def _proportional_allocation(
    counts: dict[int, int], order: list[int], budget: int
) -> dict[int, int]:
    """Largest-remainder allocation of `budget` across `order`, proportional to `counts`.

    Whenever `budget` covers every class (`budget >= len(order)`), no class is left at
    zero: any class the quota rounds down to zero is promoted to one, at the expense of
    the class currently furthest above its own quota. Deterministic given `counts`.
    """
    total = sum(counts[k] for k in order)
    n_classes = len(order)
    quotas = {k: counts[k] * budget / total for k in order}
    base = {k: int(quotas[k]) for k in order}
    remainder = budget - sum(base.values())
    ranked = sorted(order, key=lambda k: (-(quotas[k] - base[k]), k))
    for k in ranked[:remainder]:
        base[k] += 1
    if budget >= n_classes:
        for k in order:
            while base[k] == 0:
                donor = max((j for j in order if base[j] > 1), key=lambda j: base[j] - quotas[j])
                base[donor] -= 1
                base[k] += 1
    return base


def stratified_indices(labels, max_images: int | None, seed: int = 0) -> list[int]:
    """At most `max_images` indices, allocated across the classes in proportion to how many
    images each class has (largest-remainder rounding), sorted ascending.

    Every class present keeps at least one index whenever the budget covers every class
    (`max_images >= number of classes`); the images kept within each class are sampled with
    the seeded RNG, so the result is deterministic for a given `seed`. `None` (or a cap at
    or above the number of images) keeps everything.
    """
    values = np.asarray(labels)
    n = int(values.shape[0])
    if max_images is None or int(max_images) >= n:
        return list(range(n))
    budget = max(1, int(max_images))
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(values.tolist()):
        by_class[int(label)].append(i)
    order = sorted(by_class)
    counts = {k: len(by_class[k]) for k in order}
    alloc = _proportional_allocation(counts, order, budget)
    picked: list[int] = []
    for key in order:
        take_n = min(alloc[key], counts[key])
        if take_n:
            chosen = rng.choice(by_class[key], size=take_n, replace=False)
            picked.extend(int(v) for v in chosen)
    return sorted(picked)
