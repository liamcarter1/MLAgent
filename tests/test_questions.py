import pytest

from mlagent.ui.questions import ConsoleQuestioner, FormQuestioner, ScriptedQuestioner


def make_console(inputs: list[str]):
    printed: list[str] = []
    it = iter(inputs)
    q = ConsoleQuestioner(input_fn=lambda _prompt="": next(it), print_fn=printed.append)
    return q, printed


def test_console_choice_by_number_and_text():
    q, printed = make_console(["2"])
    assert q.choice("Pick", ["alpha", "beta"]) == "beta"
    assert any("1) alpha" in line for line in printed)
    q, _ = make_console(["alpha"])
    assert q.choice("Pick", ["alpha", "beta"]) == "alpha"


def test_console_choice_other_and_retry():
    q, printed = make_console(["9", "my own"])
    assert q.choice("Pick", ["alpha", "beta"], allow_other=True) == "my own"
    q, printed = make_console(["9", "1"])
    assert q.choice("Pick", ["alpha", "beta"], allow_other=False) == "alpha"
    assert any("Please" in line for line in printed)


def test_console_text_default_and_confirm():
    q, _ = make_console([""])
    assert q.text("Name?", default="demo") == "demo"
    q, _ = make_console(["n"])
    assert q.confirm("Go?") is False
    q, _ = make_console([""])
    assert q.confirm("Go?", default=True) is True


def test_scripted_records_and_exhausts():
    q = ScriptedQuestioner(["beta", "hello", "y"])
    assert q.choice("Pick", ["alpha", "beta"]) == "beta"
    assert q.text("Say") == "hello"
    assert q.confirm("Ok?") is True
    assert q.asked == ["Pick", "Say", "Ok?"]
    with pytest.raises(RuntimeError):
        q.text("more")


def test_console_number_validates_and_reprompts():
    q, printed = make_console(["abc", "50", "7"])
    assert q.number("Rows?", default=10, minimum=1, maximum=20) == 7.0
    assert sum("Please enter a number" in line for line in printed) == 2
    q, _ = make_console([""])
    assert q.number("Rows?", default=10) == 10.0


def test_scripted_number_parses_and_defaults():
    q = ScriptedQuestioner(["0.9", ""])
    assert q.number("Target?") == 0.9
    assert q.number("Rounds?", default=5) == 5.0


def make_form(answers, fallback_answers):
    notes: list[str] = []
    fallback = ScriptedQuestioner(list(fallback_answers))
    q = FormQuestioner(answers, fallback=fallback, note=notes.append)
    return q, fallback, notes


def test_form_answers_every_kind_of_question():
    q, fallback, notes = make_form(
        {
            "intake.goal": "Predict churn",
            "intake.task_type": "tabular classification",
            "intake.target_value": 0.85,
            "data.inject_quirks": True,
        },
        [],
    )
    assert q.text("Goal?", key="intake.goal") == "Predict churn"
    assert q.choice(
        "Task?", ["Tabular classification", "Image classification"], key="intake.task_type"
    ) == "Tabular classification"
    assert q.number("Target?", default=0.9, key="intake.target_value") == 0.85
    assert q.confirm("Quirks?", key="data.inject_quirks") is True
    assert q.used == ["intake.goal", "intake.task_type", "intake.target_value",
                      "data.inject_quirks"]
    assert fallback.asked == [] and notes == []


def test_form_falls_back_silently_when_the_key_is_absent():
    q, fallback, notes = make_form({}, ["typed answer"])
    assert q.text("Goal?", key="intake.goal") == "typed answer"
    assert fallback.asked == ["Goal?"] and notes == []
    assert q.used == []


def test_form_notes_and_falls_back_on_empty_or_invalid_values():
    q, fallback, notes = make_form(
        {"intake.goal": "", "intake.minutes_per_run": "soon", "clean.train_fraction": 5.0,
         "codegen.model_type": "quantum forest"},
        ["typed goal", "10", "0.7", "Random forest"],
    )
    assert q.text("Goal?", key="intake.goal") == "typed goal"
    assert q.number("Minutes?", default=10, key="intake.minutes_per_run") == 10.0
    assert q.number("Train?", minimum=0.5, maximum=0.9, key="clean.train_fraction") == 0.7
    assert q.choice("Model?", ["Random forest", "Gradient boosting"], allow_other=False,
                    key="codegen.model_type") == "Random forest"
    assert len(notes) == 4
    assert all("asking instead" in n for n in notes)
    assert q.used == []


def test_form_choice_allows_other_when_permitted():
    q, _fallback, notes = make_form({"data.hf_query": "credit card fraud"}, [])
    assert q.choice("Dataset?", ["iris", "titanic"], allow_other=True,
                    key="data.hf_query") == "credit card fraud"
    assert notes == []


def test_form_confirm_accepts_strings_and_bools():
    q, _fallback, _notes = make_form({"a": "yes", "b": False, "c": "N"}, [])
    assert q.confirm("A?", key="a") is True
    assert q.confirm("B?", key="b") is False
    assert q.confirm("C?", key="c") is False


def test_form_without_a_key_always_falls_back():
    q, fallback, notes = make_form({"intake.goal": "unused"}, ["typed"])
    assert q.text("Anything?") == "typed"
    assert fallback.asked == ["Anything?"] and notes == []


def test_console_and_scripted_accept_and_ignore_key():
    q, _printed = make_console(["2"])
    assert q.choice("Pick", ["alpha", "beta"], key="x.y") == "beta"
    s = ScriptedQuestioner(["hi", "0.5", "y"])
    assert s.text("Say", key="x.y") == "hi"
    assert s.number("Num", key="x.y") == 0.5
    assert s.confirm("Ok?", key="x.y") is True
