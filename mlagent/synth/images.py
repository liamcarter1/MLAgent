"""Synthetic shape images with optional realistic quirks for the audit stage to find.

The image counterpart of `synth/tabular.py`: five drawable shapes on a noisy coloured
background, with random size, position, rotation and colour, deterministic given a seed.
`class_imbalance`, `duplicate_fraction` and `blank_fraction` are the injected quirks that
give `audit_images` something to report, exactly as `SynthTabularConfig.quirks` does for
tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mlagent.imageset import IMAGE_SIZES, ImageSet, make_manifest

SHAPES = ("circle", "square", "triangle", "star", "cross")
MIN_IMAGES = 10
MAX_CLASSES = len(SHAPES)


@dataclass
class SynthImageConfig:
    n_images: int = 300
    image_size: int = 64
    n_classes: int = 3
    seed: int = 42
    noise: float = 0.1               # background speckle, 0 = flat, 1 = very noisy
    class_imbalance: float = 0.0     # 0 = balanced; 0.7 = the first class takes ~70%
    duplicate_fraction: float = 0.0  # share of images replaced by a copy of an earlier one
    blank_fraction: float = 0.0      # share of images replaced by a near-constant image

    def validate(self) -> None:
        if self.n_images < MIN_IMAGES:
            raise ValueError(f"n_images must be at least {MIN_IMAGES}")
        if not 2 <= self.n_classes <= MAX_CLASSES:
            raise ValueError(f"n_classes must be between 2 and {MAX_CLASSES}")
        if self.image_size not in IMAGE_SIZES:
            raise ValueError(f"image_size must be one of {IMAGE_SIZES}")
        for name in ("noise", "class_imbalance", "duplicate_fraction", "blank_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.duplicate_fraction + self.blank_fraction > 0.6:
            raise ValueError("duplicate_fraction + blank_fraction must not exceed 0.6")


def _class_shares(n_classes: int, imbalance: float) -> np.ndarray:
    """Share of images per class: uniform at imbalance 0, first class dominant at 1."""
    uniform = np.full(n_classes, 1.0 / n_classes)
    if imbalance <= 0:
        return uniform
    dominant = np.full(n_classes, (1.0 - imbalance) / max(1, n_classes - 1))
    dominant[0] = imbalance
    return uniform * (1 - imbalance) + dominant * imbalance


def _label_plan(cfg: SynthImageConfig, rng: np.random.Generator) -> np.ndarray:
    shares = _class_shares(cfg.n_classes, float(cfg.class_imbalance))
    counts = np.maximum(1, np.floor(shares * cfg.n_images).astype(int))
    while counts.sum() < cfg.n_images:
        counts[int(np.argmax(shares))] += 1
    while counts.sum() > cfg.n_images:
        counts[int(np.argmax(counts))] -= 1
    labels = np.repeat(np.arange(cfg.n_classes), counts)
    rng.shuffle(labels)
    return labels.astype(np.int64)


def _polygon(shape: str, cx: float, cy: float, radius: float, angle: float):
    """Vertices for one shape, rotated `angle` radians about its centre."""
    if shape == "square":
        base = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    elif shape == "triangle":
        base = [
            (math.cos(math.pi / 2 + k * 2 * math.pi / 3),
             math.sin(math.pi / 2 + k * 2 * math.pi / 3))
            for k in range(3)
        ]
    elif shape == "star":
        base = []
        for k in range(10):
            r = 1.0 if k % 2 == 0 else 0.45
            theta = math.pi / 2 + k * math.pi / 5
            base.append((r * math.cos(theta), r * math.sin(theta)))
    elif shape == "cross":
        t = 0.34
        base = [
            (-t, -1), (t, -1), (t, -t), (1, -t), (1, t), (t, t),
            (t, 1), (-t, 1), (-t, t), (-1, t), (-1, -t), (-t, -t),
        ]
    else:
        raise ValueError(f"{shape!r} has no polygon; draw it as an ellipse")
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (cx + radius * (x * cos_a - y * sin_a), cy + radius * (x * sin_a + y * cos_a))
        for x, y in base
    ]


def _draw_one(shape: str, size: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    from PIL import Image, ImageDraw

    background = tuple(int(v) for v in rng.integers(200, 256, size=3))
    colour = tuple(int(v) for v in rng.integers(0, 160, size=3))
    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)
    radius = float(rng.uniform(0.22, 0.38)) * size
    cx = float(rng.uniform(radius, size - radius))
    cy = float(rng.uniform(radius, size - radius))
    if shape == "circle":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)
    else:
        angle = float(rng.uniform(0, 2 * math.pi))
        draw.polygon(_polygon(shape, cx, cy, radius, angle), fill=colour)
    arr = np.asarray(img, dtype=np.int16)
    if noise > 0:
        speckle = rng.normal(0.0, 60.0 * float(noise), size=arr.shape)
        arr = arr + speckle
    return np.clip(arr, 0, 255).astype(np.uint8)


def generate(cfg: SynthImageConfig) -> ImageSet:
    """Draw `cfg.n_images` shape images, then inject the configured quirks.

    One index per class (its first occurrence in the label plan) is protected from being
    chosen as a duplicate or blank quirk target, so a rare class produced by a strong
    `class_imbalance` can never be overwritten out of existence; the requested quirk count
    is capped at however many unprotected indices remain.
    """
    cfg.validate()
    rng = np.random.default_rng(int(cfg.seed))
    class_names = sorted(SHAPES[: cfg.n_classes])
    labels = _label_plan(cfg, rng)
    size = int(cfg.image_size)
    images = np.stack(
        [_draw_one(class_names[int(k)], size, float(cfg.noise), rng) for k in labels]
    )

    n = int(cfg.n_images)
    n_blank = int(round(float(cfg.blank_fraction) * n))
    n_dup = int(round(float(cfg.duplicate_fraction) * n))
    protected = {int(np.flatnonzero(labels == k)[0]) for k in range(cfg.n_classes)}
    eligible = np.array([i for i in range(n) if i not in protected], dtype=int)
    n_quirk = min(n_blank + n_dup, len(eligible))
    quirk_targets = eligible[rng.permutation(len(eligible))[:n_quirk]]
    for i in quirk_targets[:n_blank]:
        level = int(rng.integers(40, 220))
        images[int(i)] = np.full((size, size, 3), level, dtype=np.uint8)
    for i in quirk_targets[n_blank:]:
        source = int(rng.integers(0, n))
        if source == int(i):
            source = (source + 1) % n
        images[int(i)] = images[source]
        labels[int(i)] = labels[source]

    manifest = make_manifest(
        labels,
        class_names,
        sources=[f"synthetic#{i}" for i in range(n)],
        sizes=[(size, size)] * n,
    )
    imageset = ImageSet(images=images, labels=labels, class_names=class_names,
                        manifest=manifest)
    imageset.validate()
    return imageset
