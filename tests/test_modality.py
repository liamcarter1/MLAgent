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
    # why); now that every modality it names is registered, the two must agree exactly.
    assert TEMPLATE_FOR_TASK == {t: m.template_family for m in MODALITIES for t in m.task_types}


def test_the_record_is_frozen():
    with pytest.raises(AttributeError):
        TABULAR.name = "other"
    assert isinstance(TABULAR, Modality)


def test_the_image_record_is_registered_and_points_at_the_image_modules():
    from mlagent import audit_images, cleaning_images
    from mlagent.datasources import drive_images, hf_images
    from mlagent.modality import IMAGE, MODALITIES
    from mlagent.synth import images as synth_images

    assert MODALITIES == (TABULAR, IMAGE)
    assert modality_for("image_classification") is IMAGE
    assert IMAGE.name == "image"
    assert IMAGE.data_file == "data.npz"
    assert IMAGE.template_family == "image_torch"
    assert IMAGE.profile_template == "image_common/profile.py"
    assert IMAGE.clean_template == "image_common/clean.py"
    assert IMAGE.teaching_material == "model_choices_images"
    assert IMAGE.profile_figures == ("thumbnails", "class_balance", "intensity", "class_means")
    assert IMAGE.generate is synth_images.generate
    assert IMAGE.load_drive is drive_images.load_folder
    assert IMAGE.load_hf is hf_images.load_image_dataset
    assert IMAGE.audit is audit_images.audit_images
    assert IMAGE.apply_steps is cleaning_images.apply_steps
    assert IMAGE.render_clean_py is cleaning_images.render_clean_py


def test_the_image_record_writes_and_reads_the_npz_pair(tmp_path):
    from mlagent.modality import IMAGE
    from mlagent.synth.images import SynthImageConfig, generate

    s = generate(SynthImageConfig(n_images=12, image_size=32, n_classes=2, seed=1))
    IMAGE.write_raw(s, tmp_path)
    assert (tmp_path / "data.npz").exists() and (tmp_path / "manifest.csv").exists()
    back = IMAGE.read(tmp_path)
    assert back.n_images == 12 and back.class_names == s.class_names
