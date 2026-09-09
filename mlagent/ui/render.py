"""Render assistant messages: markdown plus [[term]] markup -> HTML with clickable terms."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import markdown as _markdown

TERM_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

EXPLAIN_CALLBACK_NAME = "mlagent.explain"

CSS = """<style>
.mlagent-msg { font-family: system-ui, sans-serif; line-height: 1.5; max-width: 60rem; }
.mlagent-term { border-bottom: 1px dotted #2a7; color: #2a7; cursor: pointer; }
.mlagent-term:hover { background: rgba(34,170,119,0.12); }
.mlagent-explain {
  border-left: 3px solid #2a7; padding: 0.5rem 0.75rem; margin: 0.5rem 0;
  background: rgba(34,170,119,0.06);
}
</style>"""

COLAB_CLICK_JS = f"""<script>
(function(){{
  if (window.__mlagentClickBound) return;
  window.__mlagentClickBound = true;
  document.addEventListener('click', function(ev){{
    var el = ev.target.closest && ev.target.closest('.mlagent-term');
    if (!el) return;
    var term = el.getAttribute('data-term');
    if (window.google && window.google.colab && window.google.colab.kernel) {{
      window.google.colab.kernel.invokeFunction('{EXPLAIN_CALLBACK_NAME}', [term], {{}});
    }} else {{
      console.log('mlagent explain (no Colab kernel):', term);
    }}
  }});
}})();
</script>"""


def extract_terms(text: str) -> list[str]:
    seen: list[str] = []
    for m in TERM_RE.finditer(text):
        term = m.group(1).strip()
        if term and term not in seen:
            seen.append(term)
    return seen


def strip_terms(text: str) -> str:
    return TERM_RE.sub(lambda m: m.group(1).strip(), text)


def _term_span(m: re.Match) -> str:
    term = html.escape(m.group(1).strip(), quote=True)
    return f'<span class="mlagent-term" data-term="{term}">{term}</span>'


def _mask_code(text: str) -> tuple[str, dict[str, str]]:
    """Replace fenced and inline code spans with placeholders so term markup inside
    them is left untouched. Fenced blocks are masked first so their contents are not
    also matched by the inline-code pattern."""
    placeholders: dict[str, str] = {}

    def stash(m: re.Match) -> str:
        key = f"MLAGENTCODE{len(placeholders)}X"
        placeholders[key] = m.group(0)
        return key

    text = FENCE_RE.sub(stash, text)
    text = INLINE_CODE_RE.sub(stash, text)
    return text, placeholders


def to_html(text: str) -> str:
    masked, code_placeholders = _mask_code(text)

    # Protect term spans from the markdown processor by inserting them after conversion.
    term_placeholders: dict[str, str] = {}

    def stash_term(m: re.Match) -> str:
        key = f"MLAGENTTERM{len(term_placeholders)}X"
        term_placeholders[key] = _term_span(m)
        return key

    masked = TERM_RE.sub(stash_term, masked)
    for key, code in code_placeholders.items():
        masked = masked.replace(key, code)
    body = _markdown.markdown(masked, extensions=["fenced_code", "tables", "sane_lists"])
    for key, span in term_placeholders.items():
        body = body.replace(key, span)
    return f'{CSS}<div class="mlagent-msg">{body}</div>{COLAB_CLICK_JS}'


def _display(obj) -> None:  # separated so tests can monkeypatch
    from IPython.display import display

    display(obj)


def display_message(text: str) -> None:
    from IPython.display import HTML

    _display(HTML(to_html(text)))


FIGURE_CSS = """<style>
.mlagent-fig { margin: 0.75rem 0; max-width: 60rem; }
.mlagent-fig img { max-width: 100%; height: auto; display: block; }
.mlagent-fig figcaption {
  font-family: system-ui, sans-serif; font-size: 0.85rem; color: #52514e;
  line-height: 1.45; margin-top: 0.35rem;
}
.mlagent-fig .mlagent-figname { color: #898781; font-size: 0.75rem; }
</style>"""


def figure_html(path: Path | str, caption: str = "") -> str:
    """A PNG with its caption beneath, as one self-contained HTML block.

    The image is embedded as a data URI because a Colab cell cannot load a file from
    Google Drive by path, and the caption goes through `to_html` so `[[term]]` markup
    stays clickable.
    """
    path = Path(path)
    name = html.escape(path.name)
    if not path.exists():
        return f'{FIGURE_CSS}<div class="mlagent-fig">missing figure: {name}</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    parts = [
        FIGURE_CSS,
        '<figure class="mlagent-fig">',
        f'<img src="data:image/png;base64,{encoded}" alt="{name}">',
    ]
    if caption.strip():
        parts.append(
            f'<figcaption>{to_html(caption)}'
            f'<div class="mlagent-figname">{name}</div></figcaption>'
        )
    else:
        parts.append(f'<div class="mlagent-figname">{name}</div>')
    parts.append("</figure>")
    return "".join(parts)


def display_figure(path: Path | str, caption: str = "") -> None:
    from IPython.display import HTML

    _display(HTML(figure_html(path, caption)))
