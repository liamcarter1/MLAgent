from __future__ import annotations

import pytest

from mlagent.diagnose import (
    LABELS,
    Diagnosis,
    Proposal,
    apply_proposal,
    diagnose,
    diff_config,
    heuristic_proposals,
    meets_target,
)
from mlagent.spec import Spec
from mlagent.templates_io import load_schema, schema_for, validate_config

NESTED = load_schema("tabular_sklearn")


def spec(metric="accuracy", target=0.9) -> Spec:
    return Spec(goal="g", task_type="tabular_classification", metric=metric,
                target_value=target, data_source="synthetic", minutes_per_run=5,
                max_rounds=3, gpu="none")


def run(run_id, val, train=None, status="done", best_val=0.7, error=None, stopped_early=False,
        best_epoch=None):
    """A runs.jsonl entry plus its archived metrics, from a list of per-epoch losses."""
    train = train if train is not None else [v * 0.9 for v in val]
    epochs = [{"epoch": i + 1, "train_loss": t, "val_loss": v, "train_metric": 0.5,
               "val_metric": 0.6, "seconds": 0.1}
              for i, (t, v) in enumerate(zip(train, val, strict=False))]
    if best_epoch is None:
        best_epoch = 1 + min(range(len(val)), key=lambda i: val[i]) if val else None
    entry = {"run_id": run_id, "status": status, "best_val_metric": best_val if status == "done"
             else None, "best_epoch": best_epoch, "epochs_run": len(val), "error": error,
             "applied_diff": None, "checkpoint": f"checkpoints/run{run_id}.joblib"
             if status == "done" else None, "config": {}}
    metrics = {"status": status, "epochs": epochs, "best_epoch": best_epoch,
               "stopped_early": stopped_early}
    return entry, metrics


def history(*pairs):
    runs = [entry for entry, _m in pairs]
    by_run = {entry["run_id"]: m for entry, m in pairs}
    return runs, by_run


def test_labels_are_the_seven_from_the_spec():
    assert set(LABELS) == {"overfitting", "underfitting", "learning_rate_too_high", "plateau",
                           "failed_run", "improving", "target_met"}


def test_meets_target_respects_metric_direction():
    assert meets_target("accuracy", 0.9, 0.9) and not meets_target("accuracy", 0.89, 0.9)
    assert meets_target("rmse", 4.0, 5.0) and not meets_target("rmse", 6.0, 5.0)
    assert not meets_target("accuracy", None, 0.9)


def test_empty_history_raises():
    with pytest.raises(ValueError):
        diagnose([], {}, spec())


def test_failed_latest_run_is_failed_run():
    runs, by_run = history(run(1, [1.0, 0.8, 0.7]),
                           run(2, [1.0], status="failed", error="ValueError: boom"))
    d = diagnose(runs, by_run, spec())
    assert d.label == "failed_run" and d.evidence["error"] == "ValueError: boom"
    assert d.evidence["n_failed"] == 1 and d.latest_run_id == 2 and d.best_run_id == 1


def test_target_met_beats_curve_shape():
    runs, by_run = history(run(1, [1.0, 0.9, 1.2, 1.4], best_val=0.95))
    assert diagnose(runs, by_run, spec(target=0.9)).label == "target_met"


def test_overfitting_when_validation_rises_while_training_falls():
    val = [1.0, 0.8, 0.7, 0.72, 0.8, 0.9]
    train = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
    runs, by_run = history(run(1, val, train))
    d = diagnose(runs, by_run, spec())
    assert d.label == "overfitting"
    assert d.evidence["val_trend"] > 0 and d.evidence["train_trend"] < 0
    assert d.evidence["best_is_last"] is False


def test_underfitting_when_still_falling_at_the_end():
    val = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    runs, by_run = history(run(1, val))
    d = diagnose(runs, by_run, spec())
    assert d.label == "underfitting" and d.evidence["best_is_last"] is True


def test_improving_when_the_latest_run_beat_the_previous_best_and_is_still_falling():
    runs, by_run = history(run(1, [1.0, 0.9, 0.85], best_val=0.7),
                           run(2, [1.0, 0.8, 0.6, 0.5], best_val=0.8))
    d = diagnose(runs, by_run, spec())
    assert d.label == "improving" and d.improved is True and d.best_run_id == 2


def test_learning_rate_too_high_when_validation_oscillates():
    val = [1.0, 0.6, 1.1, 0.5, 1.2, 0.55, 1.3]
    runs, by_run = history(run(1, val, [v * 0.9 for v in val]))
    d = diagnose(runs, by_run, spec())
    assert d.label == "learning_rate_too_high" and d.evidence["oscillation"] >= 0.5


def test_plateau_when_flat():
    val = [1.0, 0.7, 0.6, 0.6, 0.6, 0.6]
    runs, by_run = history(run(1, val, [v * 0.95 for v in val]))
    assert diagnose(runs, by_run, spec()).label == "plateau"


def test_too_few_epochs_reads_as_underfitting():
    runs, by_run = history(run(1, [1.0, 0.9]))
    assert diagnose(runs, by_run, spec()).label == "underfitting"


def test_missing_metrics_archive_falls_back_to_plateau():
    runs, _by_run = history(run(1, [1.0, 0.9, 0.8]))
    assert diagnose(runs, {}, spec()).label == "plateau"


@pytest.mark.parametrize("family", ["gradient_boosting", "random_forest", "linear"])
@pytest.mark.parametrize("label", [lab for lab in LABELS if lab != "target_met"])
def test_heuristic_proposals_stay_inside_the_schema(family, label):
    flat = schema_for(NESTED, family)
    config = {k: rule.get("default") for k, rule in flat.items()}
    config["model_type"] = family
    d = Diagnosis(label=label, evidence={"error": "boom"}, latest_run_id=1, best_run_id=1,
                  improved=(label == "improving"))
    proposals = heuristic_proposals(d, config, flat)
    assert len(proposals) == 1 and proposals[0].rank == 1
    assert proposals[0].reason and proposals[0].expected in ("faster", "better", "steadier")
    new_config, diff, _notes = apply_proposal(config, proposals[0], NESTED)
    assert diff, (family, label)
    family_after = new_config["model_type"]
    assert validate_config(new_config, schema_for(NESTED, family_after)) == []


def test_target_met_has_no_heuristic():
    d = Diagnosis(label="target_met", evidence={}, latest_run_id=1, best_run_id=1)
    assert heuristic_proposals(d, {"model_type": "linear"}, schema_for(NESTED, "linear")) == []


def test_diff_config_lists_changed_added_and_removed_keys():
    assert diff_config({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == {
        "b": {"from": 2, "to": 3}, "c": {"from": None, "to": 4}}
    assert diff_config({"a": 1}, {})["a"] == {"from": 1, "to": None}


def test_apply_proposal_coerces_and_reports_the_diff():
    flat = schema_for(NESTED, "gradient_boosting")
    config = {k: rule.get("default") for k, rule in flat.items()}
    config["model_type"] = "gradient_boosting"
    proposal = Proposal(rank=1, changes={"learning_rate": 5.0, "bogus": 1, "epochs": "20"},
                        reason="r")
    new_config, diff, notes = apply_proposal(config, proposal, NESTED)
    assert new_config["learning_rate"] == 1.0 and new_config["epochs"] == 20
    assert "bogus" not in new_config
    assert diff == {"learning_rate": {"from": 0.1, "to": 1.0}, "epochs": {"from": 10, "to": 20}}
    assert any("Lowered learning_rate" in n for n in notes)
    assert any("bogus" in n for n in notes)


def test_apply_proposal_with_no_effective_change_has_an_empty_diff():
    flat = schema_for(NESTED, "linear")
    config = {k: rule.get("default") for k, rule in flat.items()}
    config["model_type"] = "linear"
    proposal = Proposal(rank=1, changes={"alpha": config["alpha"]}, reason="r")
    _new, diff, _notes = apply_proposal(config, proposal, NESTED)
    assert diff == {}


def test_family_switch_fills_the_target_family_defaults_and_keeps_common_keys():
    flat = schema_for(NESTED, "linear")
    config = {k: rule.get("default") for k, rule in flat.items()}
    config.update({"model_type": "linear", "epochs": 7, "seed": 3})
    proposal = Proposal(rank=1, changes={"model_type": "random_forest", "trees_per_epoch": 50},
                        reason="r")
    new_config, diff, _notes = apply_proposal(config, proposal, NESTED)
    assert new_config["model_type"] == "random_forest"
    assert new_config["epochs"] == 7 and new_config["seed"] == 3
    assert new_config["trees_per_epoch"] == 50 and "alpha" not in new_config
    assert validate_config(new_config, schema_for(NESTED, "random_forest")) == []
    assert diff["model_type"] == {"from": "linear", "to": "random_forest"}
    assert diff["alpha"]["to"] is None and diff["trees_per_epoch"]["from"] is None


def test_apply_proposal_rejects_an_unknown_family():
    config = {"model_type": "linear"}
    with pytest.raises(ValueError):
        apply_proposal(config, Proposal(rank=1, changes={"model_type": "svm"}, reason="r"),
                       NESTED)
