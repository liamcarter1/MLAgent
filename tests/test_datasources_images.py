from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from mlagent.datasources.drive_images import load_folder
from mlagent.datasources.hf_images import load_image_dataset


def write_image(path, size=(24, 18), colour=(10, 200, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def build_folder(root):
    for i in range(3):
        write_image(root / "cats" / f"c{i}.png", colour=(200, 10, 10))
    for i in range(2):
        write_image(root / "dogs" / f"d{i}.JPG", colour=(10, 10, 200))
    return root


def test_load_folder_reads_class_subfolders_case_insensitively(tmp_path):
    imageset, skipped = load_folder(build_folder(tmp_path), image_size=32)
    imageset.validate()
    assert imageset.n_images == 5
    assert imageset.class_names == ["cats", "dogs"]
    assert imageset.class_counts() == {"cats": 3, "dogs": 2}
    assert imageset.images.shape == (5, 32, 32, 3)
    assert skipped == []


def test_the_manifest_keeps_the_pre_resize_size_and_the_file_path(tmp_path):
    imageset, _skipped = load_folder(build_folder(tmp_path), image_size=32)
    assert (imageset.manifest["source_width"] == 24).all()
    assert (imageset.manifest["source_height"] == 18).all()
    assert imageset.manifest["source"].str.endswith((".png", ".JPG")).all()


def test_unreadable_and_root_level_files_are_skipped_not_raised(tmp_path):
    root = build_folder(tmp_path)
    (root / "cats" / "broken.png").write_bytes(b"not an image at all")
    write_image(root / "stray.png")
    (root / "notes.txt").write_text("hello", encoding="utf-8")

    imageset, skipped = load_folder(root, image_size=32)
    assert imageset.n_images == 5
    reasons = {name: reason for name, reason in skipped}
    assert any(k.endswith("broken.png") for k in reasons)
    assert any(k.endswith("stray.png") and v == "no class folder" for k, v in reasons.items())
    assert not any(k.endswith("notes.txt") for k in reasons)


def test_max_images_caps_with_a_stratified_sample(tmp_path):
    imageset, _skipped = load_folder(build_folder(tmp_path), image_size=32, max_images=4)
    assert imageset.n_images == 4
    assert set(imageset.class_counts()) == {"cats", "dogs"}
    assert min(imageset.class_counts().values()) >= 1


def test_an_empty_folder_raises_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no readable images"):
        load_folder(tmp_path, image_size=32)


class FakeClassLabel:
    def __init__(self, names):
        self.names = list(names)


class FakeImageFeature:
    pass


class FakeDataset:
    """Just enough of a `datasets.Dataset` for the loader: features, len, indexing."""

    def __init__(self, rows, features):
        self.rows = list(rows)
        self.features = dict(features)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def fake_hf_dataset(n=6):
    rows = []
    for i in range(n):
        rows.append({
            "image": Image.new("RGB", (40, 30), (i * 20 % 256, 100, 50)),
            "label": i % 2,
        })
    return FakeDataset(rows, {"image": FakeImageFeature(),
                              "label": FakeClassLabel(["cat", "dog"])})


def test_load_image_dataset_uses_the_injected_loader_and_the_class_label_names():
    calls = []

    def loader(dataset_id, split=None, **kwargs):
        calls.append((dataset_id, split, kwargs))
        return fake_hf_dataset()

    imageset = load_image_dataset("acme/pets", image_size=16, loader=loader)
    imageset.validate()
    assert calls == [("acme/pets", "train", {})]
    assert imageset.class_names == ["cat", "dog"]
    assert imageset.n_images == 6
    assert imageset.images.shape == (6, 16, 16, 3)
    assert imageset.manifest["source"].tolist()[0] == "hf:acme/pets/train#0"


def test_load_image_dataset_caps_with_a_stratified_sample():
    imageset = load_image_dataset(
        "acme/pets", image_size=16, max_images=4, loader=lambda *a, **k: fake_hf_dataset(10)
    )
    assert imageset.n_images == 4
    assert set(np.unique(imageset.labels).tolist()) == {0, 1}


def test_explicit_column_names_win_over_detection():
    rows = [{"pic": Image.new("RGB", (8, 8)), "kind": "a"} for _ in range(4)]
    ds = FakeDataset(rows, {"pic": FakeImageFeature(), "kind": FakeClassLabel(["a"])})
    imageset = load_image_dataset(
        "x/y", image_size=8, image_column="pic", label_column="kind",
        loader=lambda *a, **k: ds,
    )
    assert imageset.class_names == ["a"]
    assert imageset.n_images == 4


def test_a_dataset_with_no_image_column_raises():
    ds = FakeDataset([{"text": "hi", "label": 0}], {"text": object(),
                                                    "label": FakeClassLabel(["x"])})
    with pytest.raises(ValueError, match="image column"):
        load_image_dataset("x/y", image_size=8, loader=lambda *a, **k: ds)


def fake_int_label_dataset(values):
    """No `ClassLabel`-style `.names` on the label feature, just plain int values."""
    rows = [
        {"image": Image.new("RGB", (10, 10), (i % 256, 50, 50)), "label": v}
        for i, v in enumerate(values)
    ]
    return FakeDataset(rows, {"image": FakeImageFeature(), "label": object()})


def test_int_labels_without_class_label_names_are_mapped_through_a_numeric_index():
    ds = fake_int_label_dataset([3, 4, 5, 4, 3])
    imageset = load_image_dataset("x/y", image_size=8, loader=lambda *a, **k: ds)
    imageset.validate()
    assert imageset.class_names == ["3", "4", "5"]
    assert imageset.labels.tolist() == [0, 1, 2, 1, 0]


def test_ten_or_more_int_classes_sort_numerically_not_lexicographically():
    ds = fake_int_label_dataset(list(range(12)))
    imageset = load_image_dataset("x/y", image_size=8, loader=lambda *a, **k: ds)
    imageset.validate()
    assert imageset.class_names[2] == "2"
    assert imageset.class_names[10] == "10"
    assert imageset.labels.tolist() == list(range(12))


def test_a_dict_with_raw_image_bytes_decodes():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), (5, 5, 5)).save(buf, format="PNG")
    rows = [{"image": {"bytes": buf.getvalue()}, "label": 0}]
    ds = FakeDataset(rows, {"image": FakeImageFeature(), "label": FakeClassLabel(["a"])})
    imageset = load_image_dataset("x/y", image_size=8, loader=lambda *a, **k: ds)
    imageset.validate()
    assert imageset.n_images == 1


def test_an_undecodable_image_value_raises_naming_the_row_and_column():
    rows = [{"image": 12345, "label": 0}]
    ds = FakeDataset(rows, {"image": FakeImageFeature(), "label": FakeClassLabel(["a"])})
    with pytest.raises(ValueError, match="row 0.*'image'"):
        load_image_dataset("x/y", image_size=8, loader=lambda *a, **k: ds)
