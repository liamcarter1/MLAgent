from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mlagent.imageset import (
    MANIFEST_COLUMNS,
    ImageSet,
    blank_indices,
    duplicate_groups,
    image_hash,
    make_manifest,
    prepare_image,
    read_pair,
    stratified_indices,
    write_pair,
)


def tiny_set(n: int = 6, size: int = 8) -> ImageSet:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(n, size, size, 3), dtype=np.uint8)
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)
    class_names = ["circle", "square"]
    manifest = make_manifest(
        labels, class_names,
        sources=[f"synthetic#{i}" for i in range(n)],
        sizes=[(size * 2, size * 3)] * n,
    )
    return ImageSet(images=images, labels=labels, class_names=class_names, manifest=manifest)


def test_properties_and_class_counts():
    s = tiny_set(n=6, size=8)
    assert s.n_images == 6
    assert s.image_size == 8
    assert s.n_channels == 3
    assert s.class_counts() == {"circle": 3, "square": 3}


def test_manifest_columns_and_source_sizes():
    s = tiny_set(n=4)
    assert list(s.manifest.columns) == list(MANIFEST_COLUMNS)
    assert s.manifest["index"].tolist() == [0, 1, 2, 3]
    assert s.manifest["class_name"].tolist() == ["circle", "square", "circle", "square"]
    assert s.manifest["source_width"].tolist() == [16, 16, 16, 16]
    assert s.manifest["source_height"].tolist() == [24, 24, 24, 24]


def test_write_pair_then_read_pair_round_trips(tmp_path):
    s = tiny_set(n=5)
    npz_path, manifest_path = write_pair(s, tmp_path)
    assert npz_path.name == "data.npz" and manifest_path.name == "manifest.csv"
    back = read_pair(tmp_path)
    assert np.array_equal(back.images, s.images)
    assert np.array_equal(back.labels, s.labels)
    assert back.class_names == s.class_names
    assert back.manifest["source"].tolist() == s.manifest["source"].tolist()


def test_prepare_image_makes_a_square_rgb_array_of_the_asked_size():
    src = Image.new("L", (40, 20), color=128)
    out = prepare_image(src, 16)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8
    assert int(out.min()) == int(out.max()) == 128


def test_prepare_image_centre_crops_the_long_side():
    src = Image.new("RGB", (30, 10), color=(0, 0, 0))
    for x in range(10, 20):
        for y in range(10):
            src.putpixel((x, y), (255, 255, 255))
    out = prepare_image(src, 10)
    assert out.mean() > 250


def test_image_hash_and_duplicate_groups():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.ones((4, 4, 3), dtype=np.uint8)
    assert image_hash(a) == image_hash(a.copy())
    assert image_hash(a) != image_hash(b)
    images = np.stack([a, b, a, b, a])
    assert duplicate_groups(images) == [[0, 2, 4], [1, 3]]


def test_duplicate_groups_is_empty_when_every_image_differs():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 255, size=(5, 6, 6, 3), dtype=np.uint8)
    assert duplicate_groups(images) == []


def test_blank_indices_finds_near_constant_images():
    rng = np.random.default_rng(2)
    noisy = rng.integers(0, 255, size=(3, 8, 8, 3), dtype=np.uint8)
    flat = np.full((1, 8, 8, 3), 200, dtype=np.uint8)
    images = np.concatenate([noisy, flat])
    assert blank_indices(images) == [3]


def test_stratified_indices_caps_and_keeps_every_class():
    labels = np.array([0] * 10 + [1] * 4)
    picked = stratified_indices(labels, max_images=6, seed=0)
    assert len(picked) == 6
    assert set(labels[picked].tolist()) == {0, 1}
    assert picked == sorted(picked)


def test_stratified_indices_returns_everything_when_uncapped():
    labels = np.array([0, 1, 0, 1])
    assert stratified_indices(labels, max_images=None) == [0, 1, 2, 3]
    assert stratified_indices(labels, max_images=99) == [0, 1, 2, 3]


def test_take_keeps_the_chosen_rows_and_reindexes_the_manifest():
    s = tiny_set(n=6)
    kept = s.take([1, 3, 5])
    assert kept.n_images == 3
    assert kept.manifest["index"].tolist() == [0, 1, 2]
    assert kept.manifest["source"].tolist() == ["synthetic#1", "synthetic#3", "synthetic#5"]
    assert np.array_equal(kept.images, s.images[[1, 3, 5]])


def test_validate_rejects_a_mismatched_label_count():
    rng = np.random.default_rng(3)
    images = rng.integers(0, 255, size=(3, 4, 4, 3), dtype=np.uint8)
    labels = np.array([0, 1])
    manifest = make_manifest(labels, ["a", "b"], ["x", "y"], [(4, 4), (4, 4)])
    with pytest.raises(ValueError, match="labels"):
        ImageSet(images=images, labels=labels, class_names=["a", "b"],
                 manifest=manifest).validate()
