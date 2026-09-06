import pytest

from mlagent.ui.questions import ConsoleQuestioner, ScriptedQuestioner


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
