from __future__ import annotations

import numpy as np

from mlagent.audit_images import MIN_IMAGES_PER_CLASS, audit_images
from mlagent.imageset import ImageSet, make_manifest


def build(labels, class_names, images=None, sizes=None, image_size=16):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if images is None:
        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(n, image_size, image_size, 3), dtype=np.uint8)
    sizes = sizes or [(image_size * 4, image_size * 4)] * n
    manifest = make_manifest(labels, class_names, [f"file{i}.png" for i in range(n)], sizes)
    return ImageSet(images=images, labels=labels, class_names=class_names, manifest=manifest)


META = {"image_size": 16, "task_type": "image_classification"}


def kinds(issues):
    return [i.kind for i in issues]


def test_no_issues_on_a_balanced_varied_dataset():
    labels = [0] * 20 + [1] * 20
    assert audit_images(build(labels, ["a", "b"]), META) == []


def test_class_imbalance_fires_without_a_fix():
    labels = [0] * 95 + [1] * 5
    issues = [i for i in audit_images(build(labels, ["a", "b"]), META)
              if i.kind == "class_imbalance"]
    assert len(issues) == 1
    assert issues[0].severity == "medium"
    assert issues[0].fix is None
    assert issues[0].evidence["majority_fraction"] >= 0.9


def test_tiny_classes_are_high_severity_with_no_automatic_fix():
    labels = [0] * 30 + [1] * 3
    issues = [i for i in audit_images(build(labels, ["a", "b"]), META)
              if i.kind == "tiny_classes"]
    assert len(issues) == 1
    assert issues[0].severity == "high"
    assert issues[0].fix is None
    assert issues[0].evidence["classes"] == {"b": 3}
    assert str(MIN_IMAGES_PER_CLASS) in issues[0].message


def test_duplicate_images_propose_dropping_all_but_the_first_of_each_group():
    base = np.zeros((16, 16, 3), dtype=np.uint8)
    other = np.full((16, 16, 3), 90, dtype=np.uint8)
    rng = np.random.default_rng(1)
    unique = rng.integers(0, 255, size=(16, 16, 16, 3), dtype=np.uint8)
    images = np.concatenate([np.stack([base, base, base, other, other]), unique])
    labels = [0] * 10 + [1] * 11
    issues = [i for i in audit_images(build(labels, ["a", "b"], images=images), META)
              if i.kind == "duplicate_images"]
    assert len(issues) == 1
    assert issues[0].fix["op"] == "drop_indices"
    assert issues[0].fix["params"]["indices"] == [1, 2, 4]
    assert "duplicate" in issues[0].fix["params"]["reason"]


def test_blank_images_propose_dropping_them():
    rng = np.random.default_rng(2)
    images = rng.integers(0, 255, size=(20, 16, 16, 3), dtype=np.uint8)
    images[3] = 180
    images[11] = 20
    labels = [0] * 10 + [1] * 10
    issues = [i for i in audit_images(build(labels, ["a", "b"], images=images), META)
              if i.kind == "blank_images"]
    assert len(issues) == 1
    assert issues[0].fix["params"]["indices"] == [3, 11]


def test_upscaled_tiny_sources_are_informational_only():
    labels = [0] * 10 + [1] * 10
    sizes = [(6, 6)] * 5 + [(64, 64)] * 15
    issues = [i for i in audit_images(build(labels, ["a", "b"], sizes=sizes), META)
              if i.kind == "upscaled_sources"]
    assert len(issues) == 1
    assert issues[0].severity == "low"
    assert issues[0].fix is None
    assert issues[0].evidence["count"] == 5


def test_skipped_files_become_one_informational_issue():
    labels = [0] * 10 + [1] * 10
    skipped = [("a/broken.png", "OSError: truncated"), ("stray.png", "no class folder")]
    issues = [i for i in audit_images(build(labels, ["a", "b"]), META, skipped=skipped)
              if i.kind == "unreadable_files"]
    assert len(issues) == 1
    assert issues[0].severity == "low"
    assert issues[0].fix is None
    assert issues[0].evidence["count"] == 2
    assert "no class folder" in json_dumps(issues[0].evidence)


def json_dumps(value) -> str:
    import json

    return json.dumps(value, default=str)


def test_drop_indices_is_the_only_fix_op_ever_proposed():
    rng = np.random.default_rng(3)
    images = rng.integers(0, 255, size=(25, 16, 16, 3), dtype=np.uint8)
    images[1] = images[0]
    images[5] = 200
    labels = [0] * 22 + [1] * 3
    issues = audit_images(build(labels, ["a", "b"], images=images), META,
                          skipped=[("x.png", "bad")])
    ops = {i.fix["op"] for i in issues if i.fix}
    assert ops == {"drop_indices"}
    assert set(kinds(issues)) >= {"tiny_classes", "duplicate_images", "blank_images"}


def test_issues_are_sorted_high_severity_first():
    labels = [0] * 30 + [1] * 2
    issues = audit_images(build(labels, ["a", "b"]), META)
    severities = [i.severity for i in issues]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])
