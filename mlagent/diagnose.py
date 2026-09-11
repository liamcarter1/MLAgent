"""Read a run history and say what the curves show; propose the next change without an LLM.

Pure and deterministic. The tune stage calls `diagnose` before asking Claude for proposals,
falls back to `heuristic_proposals` when Claude is unavailable, and turns any proposal,
Claude's or ours, into a validated config plus the diff that produced it with
`apply_proposal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlagent.runlog import HIGHER_IS_BETTER, best_run, is_better
from mlagent.templates_io import coerce_config, default_config, schema_for

LABELS = (
    "overfitting",
    "underfitting",
    "learning_rate_too_high",
    "plateau",
    "failed_run",
    "improving",
    "target_met",
)
EXPECTATIONS = ("faster", "better", "steadier")

# Relative change in validation loss over the last third of the epochs that still counts
# as flat; a relative train/val gap at the best epoch that counts as overfitting; the share
# of epoch-to-epoch validation-loss changes that flip sign before we call it oscillation.
TREND_FLAT = 0.02
GAP_LARGE = 0.25
OSCILLATING = 0.5
MIN_EPOCHS_TO_JUDGE = 3
MIN_EPOCHS_FOR_OSCILLATION = 4


@dataclass
class Diagnosis:
    label: str
    evidence: dict = field(default_factory=dict)
    latest_run_id: int | None = None
    best_run_id: int | None = None
    improved: bool = False


@dataclass
class Proposal:
    rank: int
    changes: dict[str, object]
    reason: str
    expected: str = "better"


def meets_target(metric: str, value, target) -> bool:
    if not isinstance(value, int | float) or not isinstance(target, int | float):
        return False
    if isinstance(value, bool) or isinstance(target, bool):
        return False
    if HIGHER_IS_BETTER.get(metric, True):
        return value >= target
    return value <= target


def _trend(values: list[float]) -> float:
    """Relative change from the start of the last third to the end; negative is falling."""
    if len(values) < 2:
        return 0.0
    window = values[-max(2, -(-len(values) // 3)):]
    first, last = window[0], window[-1]
    return (last - first) / max(abs(first), 1e-9)


def _oscillation(values: list[float]) -> float:
    """Share of consecutive changes that flip direction; 1.0 is a perfect zigzag."""
    deltas = [b - a for a, b in zip(values, values[1:], strict=False) if b != a]
    if len(deltas) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(deltas, deltas[1:], strict=False) if (a > 0) != (b > 0))
    return flips / (len(deltas) - 1)


def diagnose(runs: list[dict], metrics_by_run: dict[int, dict], spec) -> Diagnosis:
    """One label for the latest run, with the numbers that justify it."""
    if not runs:
        raise ValueError("no runs to diagnose")
    latest = runs[-1]
    latest_id = latest.get("run_id")
    best = best_run(runs, spec.metric)
    best_id = best.get("run_id") if best else None
    previous_best = best_run([r for r in runs if r is not latest], spec.metric)
    improved = bool(
        latest.get("status") == "done"
        and previous_best is not None
        and is_better(spec.metric, latest.get("best_val_metric"),
                      previous_best.get("best_val_metric"))
    )
    evidence: dict = {
        "n_runs": len(runs),
        "n_failed": sum(1 for r in runs if r.get("status") != "done"),
        "best_val_metric": best.get("best_val_metric") if best else None,
        "previous_best": previous_best.get("best_val_metric") if previous_best else None,
        "target_value": spec.target_value,
    }
    if latest.get("status") != "done":
        evidence["error"] = str(latest.get("error") or "unknown error")
        return Diagnosis("failed_run", evidence, latest_id, best_id, False)
    if best is not None and meets_target(spec.metric, best.get("best_val_metric"),
                                         spec.target_value):
        return Diagnosis("target_met", evidence, latest_id, best_id, improved)

    metrics = metrics_by_run.get(latest_id) or {}
    epochs = [e for e in (metrics.get("epochs") or []) if isinstance(e, dict)]
    val = [float(e["val_loss"]) for e in epochs if e.get("val_loss") is not None]
    train = [float(e["train_loss"]) for e in epochs if e.get("train_loss") is not None]
    n = len(val)
    best_epoch = latest.get("best_epoch") or n
    evidence.update({
        "n_epochs": n,
        "best_epoch": best_epoch,
        "best_is_last": bool(n) and best_epoch == n,
        "stopped_early": bool(metrics.get("stopped_early")),
    })
    if n == 0:
        evidence["no_epoch_data"] = True
        return Diagnosis("plateau", evidence, latest_id, best_id, improved)
    if n < MIN_EPOCHS_TO_JUDGE:
        evidence["val_trend"] = round(_trend(val), 4)
        label = "improving" if improved else "underfitting"
        return Diagnosis(label, evidence, latest_id, best_id, improved)

    val_trend, train_trend, osc = _trend(val), _trend(train), _oscillation(val)
    i = min(max(int(best_epoch) - 1, 0), n - 1)
    gap = (val[i] - train[i]) / max(abs(val[i]), 1e-9) if i < len(train) else 0.0
    evidence.update({
        "val_trend": round(val_trend, 4),
        "train_trend": round(train_trend, 4),
        "gap": round(gap, 4),
        "oscillation": round(osc, 4),
    })
    if osc >= OSCILLATING and n >= MIN_EPOCHS_FOR_OSCILLATION:
        label = "learning_rate_too_high"
    elif (val_trend > TREND_FLAT and train_trend < 0) or (gap > GAP_LARGE and best_epoch < n):
        label = "overfitting"
    elif best_epoch == n and val_trend < -TREND_FLAT:
        label = "improving" if improved else "underfitting"
    else:
        label = "plateau"
    return Diagnosis(label, evidence, latest_id, best_id, improved)


IMAGE_FAMILIES = ("tiny_cnn", "small_cnn", "resnet18")


def _image_move(label: str, choose, scaled, bumped) -> tuple[str, str]:
    """The fixed image-family move per diagnosis label; returns (expected, reason)."""
    if label == "overfitting":
        choose("augment", "basic")
        bumped("dropout", 1.4, 0.2)
        bumped("weight_decay", 10.0, 0.001)
        return "steadier", (
            "Validation [[loss]] turned upward while training loss kept falling: the "
            "network is memorising these exact pictures. Random flips and shifts, more "
            "[[dropout]] and a stronger weight penalty make that much harder."
        )
    if label in ("underfitting", "improving"):
        scaled("epochs", 1.5)
        scaled("learning_rate", 1.5)
        reason = (
            "The last change helped and validation loss was still falling when training "
            "stopped, so keep going in the same direction with more [[epoch]]s and a "
            "bigger step each time."
            if label == "improving"
            else "Validation loss was still falling at the last [[epoch]]: the network had "
                 "not finished learning. More epochs and a bigger step per update let it."
        )
        return "better", reason
    if label == "learning_rate_too_high":
        scaled("learning_rate", 0.3)
        scaled("batch_size", 2)
        return "steadier", (
            "Validation loss jumps up and down instead of settling: each update "
            "overshoots. A much smaller [[learning rate]] takes smaller steps, and a "
            "bigger batch averages out more of the noise before each one."
        )
    if label == "plateau":
        scaled("learning_rate", 0.5)
        return "better", (
            "The curve has flattened. Halving the [[learning rate]] lets the network "
            "settle into a better spot instead of bouncing over it."
        )
    scaled("learning_rate", 0.3)
    scaled("epochs", 0.5)
    return "steadier", (
        "The run failed, most often from the loss blowing up to infinity. A much smaller "
        "[[learning rate]] and fewer epochs are the gentlest retry."
    )


def heuristic_proposals(diagnosis: Diagnosis, config: dict, schema: dict) -> list[Proposal]:
    """One fixed proposal per label and family, used when Claude is unavailable.

    Every helper checks the flat `schema` before setting a key, so a proposal for one
    family can never set another family's key (an image proposal never reaches for
    `min_samples_leaf`, a tabular one never reaches for `augment`). Values are scaled from
    the current config; `apply_proposal` clamps them to the schema.
    """
    label = diagnosis.label
    family = str(config.get("model_type", ""))
    changes: dict[str, object] = {}
    expected = "better"

    def scaled(key: str, factor: float) -> None:
        if key not in schema:
            return
        current = config.get(key)
        if isinstance(current, int | float) and not isinstance(current, bool):
            changes[key] = current * factor

    def bumped(key: str, factor: float, floor: float) -> None:
        """Scale a key that may sit at zero, so it still moves off the floor."""
        if key not in schema:
            return
        current = config.get(key)
        if not isinstance(current, int | float) or isinstance(current, bool):
            return
        changes[key] = current * factor if current else floor

    def choose(key: str, value: str) -> None:
        if key in schema and config.get(key) != value:
            changes[key] = value

    if label == "target_met":
        return []

    if family in IMAGE_FAMILIES:
        expected, reason = _image_move(label, choose, scaled, bumped)
    elif label == "overfitting":
        expected = "steadier"
        if family == "gradient_boosting":
            scaled("min_samples_leaf", 2)
            scaled("learning_rate", 0.5)
            reason = ("Validation [[loss]] turned upward while training loss kept falling: "
                      "the trees are fitting noise. Larger leaves and a smaller "
                      "[[learning rate]] make each tree more cautious.")
        elif family == "random_forest":
            scaled("min_samples_leaf", 3)
            scaled("max_features", 0.5)
            reason = ("The forest is memorising rows. Requiring more rows per leaf and "
                      "showing each tree fewer columns makes the trees disagree in useful "
                      "ways instead of all copying the noise.")
        else:
            bumped("alpha", 10.0, 0.001)
            reason = ("A linear model overfits when its weights grow large; a stronger "
                      "[[regularisation]] penalty (`alpha`) keeps them small.")
    elif label in ("underfitting", "improving"):
        scaled("epochs", 1.5)
        if family == "gradient_boosting":
            scaled("learning_rate", 1.5)
        elif family == "random_forest":
            scaled("trees_per_epoch", 2)
        reason = (
            "The last change helped and validation loss was still falling when training "
            "stopped, so keep going in the same direction with more epochs."
            if label == "improving"
            else "Validation loss was still falling at the last [[epoch]]: the model had not "
                 "finished learning. More epochs (and a bigger step per epoch) let it."
        )
    elif label == "learning_rate_too_high":
        expected = "steadier"
        if family == "random_forest":
            scaled("trees_per_epoch", 2)
            reason = ("A forest has no learning rate; the validation loss jumps because each "
                      "epoch adds too few trees to average out the noise. Doubling the trees "
                      "per epoch smooths it.")
        else:
            scaled("learning_rate", 0.3)
            reason = ("Validation loss jumps up and down instead of settling: each update "
                      "overshoots. A much smaller [[learning rate]] takes smaller steps.")
    elif label == "plateau":
        if family == "gradient_boosting":
            scaled("max_leaf_nodes", 2)
            scaled("learning_rate", 0.5)
            reason = ("The curve has flattened. Bigger trees can learn finer structure; a "
                      "smaller learning rate stops them overshooting while they do.")
        elif family == "random_forest":
            scaled("max_features", 1.5)
            scaled("trees_per_epoch", 2)
            reason = ("The forest has flattened. Letting each tree see more columns and "
                      "adding more trees per epoch gives it more to average over.")
        else:
            changes["model_type"] = "gradient_boosting"
            reason = ("A linear model has flattened out: it cannot learn interactions "
                      "between columns. [[Gradient boosting]] can, so switch family.")
    else:  # failed_run
        expected = "steadier"
        if family == "random_forest":
            scaled("trees_per_epoch", 0.5)
            reason = ("The run failed; fewer trees per epoch make each step cheaper and "
                      "get past most memory or timeout failures.")
        else:
            scaled("learning_rate", 0.3)
            scaled("epochs", 0.5)
            reason = ("The run failed, most often from losses blowing up. A much smaller "
                      "[[learning rate]] and fewer epochs are the gentlest retry.")
    if not changes:
        return []
    return [Proposal(rank=1, changes=changes, reason=reason, expected=expected)]


def diff_config(old: dict, new: dict) -> dict[str, dict]:
    """`{key: {"from": old, "to": new}}` for every key whose value differs, in old's order."""
    diff: dict[str, dict] = {}
    for key in [*old, *(k for k in new if k not in old)]:
        before, after = old.get(key), new.get(key)
        if before != after:
            diff[key] = {"from": before, "to": after}
    return diff


def apply_proposal(
    config: dict, proposal: Proposal, nested_schema: dict
) -> tuple[dict, dict, list[str]]:
    """Coerce a proposal against the (possibly new) family's schema; return the config, the
    diff from the current config, and the coercion notes. Raises ValueError on an unknown
    `model_type`."""
    changes = dict(proposal.changes or {})
    family = str(changes.get("model_type") or config.get("model_type") or "")
    flat = schema_for(nested_schema, family)
    if family != config.get("model_type"):
        base = default_config(flat)
        for key in nested_schema.get("common") or {}:
            if key in config and key != "model_type":
                base[key] = config[key]
    else:
        base = dict(config)
    merged = {**base, **changes, "model_type": family}
    new_config, notes = coerce_config(merged, flat)
    return new_config, diff_config(config, new_config), notes
