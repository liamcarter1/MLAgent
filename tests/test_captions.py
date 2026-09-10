from __future__ import annotations

from pathlib import Path

from mlagent.captions import CAPTIONS, caption_for

EXPECTED_KINDS = {
    "histograms", "missing", "class_balance", "target_distribution", "correlation",
    "clean_before_after_missing", "training_curves", "confusion", "roc_pr", "per_class",
    "pred_vs_actual", "residuals", "compare_curves", "compare_runs",
    "thumbnails", "intensity", "class_means", "misclassified", "clean_before_after_classes",
}


def test_every_figure_kind_has_a_caption():
    assert set(CAPTIONS) == EXPECTED_KINDS
    for kind, text in CAPTIONS.items():
        assert text.strip(), kind
        assert len(text) < 400, kind


def test_caption_for_matches_by_filename_suffix():
    assert caption_for(Path("plots/raw_histograms.png")) == CAPTIONS["histograms"]
    assert caption_for("plots/clean_histograms.png") == CAPTIONS["histograms"]
    assert caption_for(Path("plots/run1_val_confusion.png")) == CAPTIONS["confusion"]
    assert caption_for(Path("plots/test_roc_pr.png")) == CAPTIONS["roc_pr"]
    assert caption_for(Path("plots/training_curves.png")) == CAPTIONS["training_curves"]
    assert caption_for(Path("plots/run3_training.png")) == CAPTIONS["training_curves"]


def test_longest_suffix_wins_and_unknown_returns_empty():
    # "clean_before_after_missing" also ends in "missing"; the longer key must win.
    assert caption_for("plots/clean_before_after_missing.png") == (
        CAPTIONS["clean_before_after_missing"]
    )
    assert caption_for("plots/something_else.png") == ""


def test_comparison_captions_match_by_stem():
    assert caption_for("plots/compare_curves.png") == CAPTIONS["compare_curves"]
    assert caption_for("plots/compare_runs.png") == CAPTIONS["compare_runs"]


def test_the_image_caption_kinds_exist_and_read_as_guidance():
    from mlagent.captions import CAPTIONS

    for kind in ("thumbnails", "intensity", "class_means", "misclassified",
                 "clean_before_after_classes"):
        assert kind in CAPTIONS
        assert len(CAPTIONS[kind]) > 80


def test_caption_for_matches_the_new_image_figure_names():
    from mlagent.captions import CAPTIONS, caption_for

    assert caption_for("raw_thumbnails.png") == CAPTIONS["thumbnails"]
    assert caption_for("clean_class_means.png") == CAPTIONS["class_means"]
    assert caption_for("val_misclassified.png") == CAPTIONS["misclassified"]
    assert caption_for("raw_intensity.png") == CAPTIONS["intensity"]
    assert caption_for("clean_before_after_classes.png") == (
        CAPTIONS["clean_before_after_classes"]
    )
