from __future__ import annotations

import numpy as np
import pytest

from mlagent.imageset import blank_indices, duplicate_groups
from mlagent.synth.images import SHAPES, SynthImageConfig, generate


def test_generate_returns_a_valid_imageset_of_the_asked_shape():
    cfg = SynthImageConfig(n_images=40, image_size=32, n_classes=3, seed=7)
    s = generate(cfg)
    s.validate()
    assert s.n_images == 40
    assert s.images.shape == (40, 32, 32, 3)
    assert s.images.dtype == np.uint8
    assert s.class_names == sorted(SHAPES[:3])
    assert set(np.unique(s.labels).tolist()) <= set(range(3))


def test_the_manifest_records_the_source_size_before_resizing():
    s = generate(SynthImageConfig(n_images=12, image_size=32, n_classes=2, seed=1))
    assert s.manifest["source"].str.startswith("synthetic#").all()
    assert (s.manifest["source_width"] == 32).all()
    assert (s.manifest["source_height"] == 32).all()


def test_generation_is_deterministic_given_a_seed():
    cfg = SynthImageConfig(n_images=20, image_size=32, n_classes=3, seed=99)
    a, b = generate(cfg), generate(cfg)
    assert np.array_equal(a.images, b.images)
    assert np.array_equal(a.labels, b.labels)
    other = generate(SynthImageConfig(n_images=20, image_size=32, n_classes=3, seed=100))
    assert not np.array_equal(a.images, other.images)


def test_the_shapes_are_actually_different_between_classes():
    s = generate(SynthImageConfig(n_images=60, image_size=32, n_classes=3, seed=3, noise=0.0))
    means = [s.images[s.labels == k].mean() for k in range(3)]
    assert len({round(float(m), 1) for m in means}) >= 2


def test_injected_duplicates_are_findable_by_the_audit_helper():
    s = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=5,
                                  duplicate_fraction=0.25))
    groups = duplicate_groups(s.images)
    assert sum(len(g) - 1 for g in groups) >= 8


def test_injected_blanks_are_findable_by_the_audit_helper():
    s = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=2, seed=5,
                                  blank_fraction=0.2, noise=0.0))
    assert len(blank_indices(s.images)) >= 6


def test_class_imbalance_makes_the_first_class_dominate():
    balanced = generate(SynthImageConfig(n_images=60, image_size=32, n_classes=3, seed=2))
    counts = list(balanced.class_counts().values())
    assert max(counts) - min(counts) <= 1

    skewed = generate(SynthImageConfig(n_images=60, image_size=32, n_classes=3, seed=2,
                                       class_imbalance=0.7))
    top = max(skewed.class_counts().values())
    assert top / skewed.n_images >= 0.6


def test_no_quirks_means_no_duplicates_and_no_blanks():
    s = generate(SynthImageConfig(n_images=40, image_size=32, n_classes=3, seed=11))
    assert duplicate_groups(s.images) == []
    assert blank_indices(s.images) == []


def test_strong_imbalance_with_duplicates_and_blanks_never_erases_a_rare_class():
    # duplicate_fraction + blank_fraction is capped at 0.6 by validate(); 0.5 + 0.1 sits
    # at that boundary so the quirk pool is as large as possible while staying valid.
    n_classes = 5
    for seed in range(5):
        cfg = SynthImageConfig(
            n_images=15, image_size=32, n_classes=n_classes, seed=seed,
            class_imbalance=0.9, duplicate_fraction=0.5, blank_fraction=0.1,
        )
        s = generate(cfg)
        s.validate()
        counts = np.bincount(s.labels, minlength=n_classes)
        assert counts.min() >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_images": 3},
        {"n_classes": 1},
        {"n_classes": 9},
        {"image_size": 17},
        {"noise": 2.0},
        {"duplicate_fraction": 1.5},
    ],
)
def test_validate_rejects_impossible_configs(kwargs):
    with pytest.raises(ValueError):
        generate(SynthImageConfig(**kwargs))
