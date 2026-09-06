"""Generate notebooks/ML_Training_Agent.ipynb. Run: python scripts/build_notebook.py"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "ML_Training_Agent.ipynb"

cells = [
    nbf.v4.new_markdown_cell(
        "# ML Training Agent\n\n"
        "Copy this notebook once per project. Before running:\n\n"
        "1. Copy the `mlagent/` folder from this repo to `MyDrive/ml_agent/mlagent/`.\n"
        "2. Add your Anthropic key in Colab **Secrets** (key icon) as `ANTHROPIC_API_KEY` with "
        "notebook access on.\n"
        "3. Run the cells in order. Click any highlighted term in the assistant's messages to "
        "get an explanation."
        "\n\n**Tip:** while a cell is waiting for you to type an answer, clicking a highlighted "
        "term does nothing until that cell finishes. Click terms after a stage completes, or run "
        "`colab.explain('term')` in its own cell."
    ),
    nbf.v4.new_code_cell("%pip -q install anthropic markdown"),
    nbf.v4.new_code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "import sys\n"
        "sys.path.insert(0, '/content/drive/MyDrive/ml_agent')\n"
        "from mlagent import colab\n"
        "colab.setup()"
    ),
    nbf.v4.new_code_cell(
        "PROJECT_NAME = 'my_first_project'  # change per project\n"
        "orch = colab.start(PROJECT_NAME)\n"
        "orch.run()"
    ),
    nbf.v4.new_markdown_cell("### Ask about any term"),
    nbf.v4.new_code_cell("colab.explain('validation set')"),
    nbf.v4.new_markdown_cell(
        "### Redo a stage\nResets that stage and everything after it, then reruns."
    ),
    nbf.v4.new_code_cell("# orch.reset('intake'); orch.run()"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
