"""Generate notebooks/ML_Training_Agent.ipynb. Run: python scripts/build_notebook.py"""

from pathlib import Path

import nbformat as nbf

from mlagent.colab import script_cells

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "ML_Training_Agent.ipynb"


def code(*lines: str):
    return nbf.v4.new_code_cell("\n".join(lines))


def script_cell(stage: str, index: int):
    return code(script_cells(stage)[index])


cells = [
    nbf.v4.new_markdown_cell(
        "# ML Training Agent\n\n"
        "Copy this notebook once per project. Before running:\n\n"
        "1. Copy the `mlagent/` folder from this repo to `MyDrive/ml_agent/mlagent/`.\n"
        "2. Add your Anthropic key in Colab **Secrets** (key icon) as `ANTHROPIC_API_KEY` "
        "with notebook access on.\n"
        "3. Run the cells in order.\n\n"
        "**How it works.** Cells alternate. An *assistant cell* (`orch.run(...)`) asks you "
        "questions, writes real Python scripts into your project folder on Drive, and "
        "explains them. It then stops and names the *script cell* to run next. A script "
        "cell starts as `%load train.py`: run it once to pull the code into the cell so you "
        "can read (and edit) it, then run it again to execute it. When it finishes, go back "
        "to the assistant cell and run it again for the debrief.\n\n"
        "Click any highlighted term in the assistant's messages for an explanation. While a "
        "cell is waiting for you to type an answer, clicking does nothing — click after the "
        "cell finishes, or run `colab.explain('term')` in its own cell."
    ),
    code(
        "%pip -q install anthropic markdown pandas numpy scikit-learn matplotlib pyarrow "
        "openpyxl huggingface_hub datasets"
    ),
    code(
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "import sys",
        "sys.path.insert(0, '/content/drive/MyDrive/ml_agent')",
        "from mlagent import colab",
        "colab.setup()",
    ),
    code(
        "#@title 1. Project and interview  { display-mode: 'form' }",
        "PROJECT_NAME = 'my_first_project'  #@param {type:'string'}",
        "LEARNING_LEVEL = 'Intermediate - explain the key ideas'  #@param "
        "['Beginner - explain everything as we go', 'Intermediate - explain the key ideas', "
        "'Expert - just the numbers']",
        "GOAL = 'Predict which customers churn'  #@param {type:'string'}",
        "TASK = 'Tabular classification'  #@param ['Tabular classification', "
        "'Tabular regression']",
        "METRIC = 'accuracy'  #@param ['accuracy', 'f1', 'rmse', 'mae', 'r2']",
        "TARGET_VALUE = 0.9  #@param {type:'number'}",
        "DATA_SOURCE = 'Synthetic data'  #@param ['Synthetic data', "
        "'Upload or Google Drive path', 'HuggingFace Hub dataset']",
        "MINUTES_PER_RUN = 10  #@param {type:'integer'}",
        "MAX_ROUNDS = 5  #@param {type:'integer'}",
        "GPU = 'No GPU (CPU only)'  #@param ['No GPU (CPU only)', 'T4 GPU', "
        "'Any available GPU']",
        "",
        "orch = colab.start(PROJECT_NAME)",
        "orch.run(until='intake', answers={",
        "    'intake.goal': GOAL,",
        "    'intake.learning_level': LEARNING_LEVEL,",
        "    'intake.task_type': TASK,",
        "    'intake.metric': METRIC,",
        "    'intake.target_value': TARGET_VALUE,",
        "    'intake.data_source': DATA_SOURCE,",
        "    'intake.minutes_per_run': MINUTES_PER_RUN,",
        "    'intake.max_rounds': MAX_ROUNDS,",
        "    'intake.gpu': GPU,",
        "})",
    ),
    code(
        "#@title 2. Data  { display-mode: 'form' }",
        "N_ROWS = 1000  #@param {type:'integer'}",
        "N_FEATURES = 8  #@param {type:'integer'}",
        "N_CLASSES = 2  #@param {type:'integer'}",
        "CLASS_BALANCE = 0.5  #@param {type:'number'}",
        "NOISE = 0.1  #@param {type:'number'}",
        "INJECT_QUIRKS = True  #@param {type:'boolean'}",
        "DRIVE_PATH = ''  #@param {type:'string'}",
        "HF_QUERY = ''  #@param {type:'string'}",
        "TARGET_COLUMN = ''  #@param {type:'string'}",
        "",
        "orch.run(until='data', answers={",
        "    'data.n_rows': N_ROWS,",
        "    'data.n_features': N_FEATURES,",
        "    'data.n_classes': N_CLASSES,",
        "    'data.class_balance': CLASS_BALANCE,",
        "    'data.noise': NOISE,",
        "    'data.inject_quirks': INJECT_QUIRKS,",
        "    'data.drive_path': DRIVE_PATH,",
        "    'data.hf_query': HF_QUERY,",
        "    'data.target_column': TARGET_COLUMN,",
        "})",
    ),
    script_cell("data", 0),
    code(
        "#@title 3. Clean  { display-mode: 'form' }",
        "DROP_COLUMNS = ''  #@param {type:'string'}",
        "TRAIN_FRACTION = 0.7  #@param {type:'number'}",
        "VAL_FRACTION = 0.15  #@param {type:'number'}",
        "",
        "# Each proposed fix is confirmed one at a time in the output below.",
        "orch.run(until='clean', answers={",
        "    'clean.drop_columns': DROP_COLUMNS,",
        "    'clean.train_fraction': TRAIN_FRACTION,",
        "    'clean.val_fraction': VAL_FRACTION,",
        "})",
    ),
    script_cell("clean", 0),
    code(
        "#@title 4. Model  { display-mode: 'form' }",
        "MODEL = 'Ask me after the explanation'  #@param "
        "['Ask me after the explanation', 'Linear / logistic regression', 'Random forest', "
        "'Gradient boosting']",
        "",
        "orch.run(until='train', answers={'codegen.model_type': MODEL})",
    ),
    script_cell("train", 0),
    script_cell("train", 1),
    code(
        "#@title 5. Report",
        "# Asks before touching the held-out test set; answer in the box below.",
        "orch.run(until='report')",
    ),
    script_cell("report", 0),
    code("orch.run()"),
    nbf.v4.new_markdown_cell(
        "## Train again\n\n"
        "Edit `config.json` in the project folder, then run the cell below: it resets the "
        "train stage and re-prepares it, naming the `train.py` and `evaluate.py` cells "
        "above for you to run again. Run those two cells, then run the `orch.run()` cell "
        "above (just before this section) once more to log the new run and rewrite the "
        "report. `orch.waiting()` says which cells the assistant is still waiting on; "
        "`orch.debrief('train')` forces the debrief if Drive's timestamps lag."
    ),
    code("orch.reset('train')", "orch.run()"),
    nbf.v4.new_markdown_cell("### Ask about any term"),
    code("colab.explain('validation set')"),
    nbf.v4.new_markdown_cell(
        "### Redo a stage\nResets that stage and everything after it, then reruns."
    ),
    code("# orch.reset('intake'); orch.run()"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
