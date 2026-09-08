"""Fixed 'how to read this chart' text, one entry per figure kind.

The agent adds a sentence about *this* dataset below expert level (see `teaching.py`);
these captions are the part that never changes and never needs an LLM call.
"""

from __future__ import annotations

from pathlib import Path

CAPTIONS: dict[str, str] = {
    "histograms": (
        "One [[histogram]] per numeric column: the x axis is the value, the height is how "
        "many rows fall in that bucket. Look for columns that are all one value, long tails "
        "on one side ([[skew]]), or two separate humps."
    ),
    "missing": (
        "Each row of the chart is a column of your data and each pixel across is a row; dark "
        "means the value is missing. Solid dark bands mean a column is mostly empty; vertical "
        "stripes mean whole rows are missing values together."
    ),
    "class_balance": (
        "How many rows carry each label. A large gap between the bars is [[class imbalance]]: "
        "a model can score well just by always predicting the biggest class, so accuracy alone "
        "will flatter it."
    ),
    "target_distribution": (
        "The spread of the value you are predicting. Check the range is what you expect and "
        "watch for a long tail: a few extreme targets pull [[regression]] errors around."
    ),
    "correlation": (
        "How strongly each pair of numeric columns moves together, from -1 (opposite) through "
        "0 (unrelated) to +1 (identical). A column almost perfectly correlated with the target "
        "is often [[data leakage]]."
    ),
    "clean_before_after_missing": (
        "The percentage of missing values per column before and after cleaning. Bars that "
        "shrink to zero were filled or dropped; bars that did not move were left alone on "
        "purpose."
    ),
    "training_curves": (
        "Left: [[loss]] per [[epoch]] for the training and validation splits. Right: the same "
        "for your chosen metric. Training loss falling while validation loss rises is "
        "[[overfitting]]; both flat is a [[plateau]]."
    ),
    "confusion": (
        "Rows are the true label, columns are what the model predicted, so the diagonal is "
        "correct. A bright off-diagonal cell names the two classes the model keeps confusing."
    ),
    "roc_pr": (
        "Left: the [[ROC curve]] — true positives against false positives as the decision "
        "threshold moves; further above the dashed line is better. Right: [[precision]] "
        "against [[recall]], which is the more honest view when classes are imbalanced."
    ),
    "per_class": (
        "Precision and recall for each class. Precision is how often a prediction of that "
        "class is right; recall is how much of that class the model finds. Small classes with "
        "low bars are the ones to fix."
    ),
    "pred_vs_actual": (
        "Each point is one row: actual value across, predicted value up. Perfect predictions "
        "sit on the dashed diagonal. Points bending away from it at one end mean the model is "
        "biased in that part of the range."
    ),
    "residuals": (
        "Left: the spread of actual minus predicted; a bell centred on zero is what you want. "
        "Right: the same errors against the prediction; a funnel or a curve means the model is "
        "missing structure rather than just being noisy."
    ),
}


# The train stage archives `training_curves.png` as `run{N}_training.png`; the shortened
# suffix maps back to the same caption.
ALIASES = {"training": "training_curves"}


def caption_for(path: Path | str) -> str:
    """The caption for a figure, matched on the longest key that ends its filename stem."""
    stem = Path(path).stem
    best = ""
    for kind in CAPTIONS:
        if (stem == kind or stem.endswith(f"_{kind}")) and len(kind) > len(best):
            best = kind
    if not best:
        best = ALIASES.get(stem.rsplit("_", 1)[-1], "")
    return CAPTIONS.get(best, "")
