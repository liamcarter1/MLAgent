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


def test_to_html_renders_fenced_code_without_linkifying_inside():
    text = "Slice it:\n\n```python\ndata[[0, 1]]\n    more = 1\n```\n"
    html = render.to_html(text)
    assert "<pre><code" in html
    assert "data[[0, 1]]" in html
    assert "    more = 1" in html
    assert 'data-term="0, 1"' not in html
    assert "mlagent-term" not in html.split("<pre>")[1].split("</pre>")[0]


def test_to_html_does_not_linkify_inline_code():
    html = render.to_html("Use `[[x]]` here and a [[real term]] outside code.")
    assert "<code>[[x]]</code>" in html
    assert 'data-term="x"' not in html
    assert '<span class="mlagent-term" data-term="real term">real term</span>' in html


def test_colab_click_js_uses_explain_callback_name_constant():
    assert render.EXPLAIN_CALLBACK_NAME in render.COLAB_CLICK_JS
