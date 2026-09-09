from __future__ import annotations

from pathlib import Path

from mlagent.captions import CAPTIONS, caption_for

EXPECTED_KINDS = {
    "histograms", "missing", "class_balance", "target_distribution", "correlation",
    "clean_before_after_missing", "training_curves", "confusion", "roc_pr", "per_class",
    "pred_vs_actual", "residuals",
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
