from mlagent.ui import render


def test_extract_and_strip_terms():
    text = "Lower the [[learning rate]] to stop [[overfitting]]; [[learning rate]] again."
    assert render.extract_terms(text) == ["learning rate", "overfitting"]
    assert render.strip_terms(text) == "Lower the learning rate to stop overfitting; learning rate again."


def test_to_html_marks_terms_and_renders_markdown():
    html = render.to_html("**Bold** and a [[batch size]] term\n\n- item")
    assert '<span class="mlagent-term" data-term="batch size">batch size</span>' in html
    assert "<strong>Bold</strong>" in html
    assert "<li>item</li>" in html
    assert "mlagent.explain" in html  # click hook present
    assert "<style>" in html


def test_to_html_escapes_html_in_terms():
    html = render.to_html("[[<b>x</b>]]")
    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def test_display_message_calls_ipython(monkeypatch):
    shown: list = []
    monkeypatch.setattr(render, "_display", lambda obj: shown.append(obj))
    render.display_message("hello [[epoch]]")
    assert len(shown) == 1
    assert "epoch" in shown[0].data
