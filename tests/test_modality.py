from __future__ import annotations

import pandas as pd
import pytest

from mlagent import audit, cleaning
from mlagent.datasources import drive, hf
from mlagent.modality import MODALITIES, TABULAR, Modality, modality_for
from mlagent.synth import tabular
from mlagent.templates_io import TEMPLATE_FOR_TASK


def test_lookup_by_task_type_returns_the_tabular_record():
    for task in ("tabular_classification", "tabular_regression"):
        m = modality_for(task)
        assert m is TABULAR
        assert m.name == "tabular"
        assert m.data_file == "data.csv"
        assert m.template_family == "tabular_sklearn"
        assert m.profile_template == "common/profile.py"
        assert m.clean_template == "common/clean.py"
        assert m.teaching_material == "model_choices"


def test_unknown_task_type_names_itself_and_the_valid_ones():
    with pytest.raises(ValueError) as exc:
        modality_for("audio_classification")
    message = str(exc.value)
    assert "audio_classification" in message
    assert "tabular_classification" in message


def test_the_tabular_record_wraps_the_existing_functions_unchanged():
    assert TABULAR.generate is tabular.generate
    assert TABULAR.load_drive is drive.load_table
    assert TABULAR.load_hf is hf.load_tabular
    assert TABULAR.audit is audit.audit_tabular
    assert TABULAR.apply_steps is cleaning.apply_steps
    assert TABULAR.render_clean_py is cleaning.render_clean_py


def test_the_tabular_record_writes_and_reads_a_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    TABULAR.write_raw(df, tmp_path)
    assert (tmp_path / "data.csv").exists()
    pd.testing.assert_frame_equal(TABULAR.read(tmp_path), df)


def test_template_for_task_agrees_with_the_registry():
    # TEMPLATE_FOR_TASK is a literal dict (see mlagent/modality.py's module docstring for
    # why) and may name a task type — "image_classification" — before that modality's
    # record is registered in MODALITIES; every task type the registry *does* know about
    # must still agree with it.
    for modality in MODALITIES:
        for task in modality.task_types:
            assert TEMPLATE_FOR_TASK[task] == modality.template_family


def test_the_record_is_frozen():
    with pytest.raises(AttributeError):
        TABULAR.name = "other"
    assert isinstance(TABULAR, Modality)
