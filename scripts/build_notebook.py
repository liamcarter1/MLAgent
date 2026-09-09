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
        "## How this notebook works\n\n"
        "The assistant leads you through six steps. Steps 1-4 each have a form cell you "
        "fill in. Step 5 (train) is two script cells you run: `train.py`, then "
        "`evaluate.py`. Step 6 (report) is a cell you just run. Some steps also hand you a "
        "script cell to run:\n\n"
        "1. **Interview.** *You do:* answer the form below (project name, goal, task, metric, "
        "data source, budget). *You get:* a saved project specification, `spec.json`.\n"
        "2. **Data.** *You do:* give the path, search keywords, or the size of the "
        "synthetic dataset (the source itself is chosen in step 1). *You get:* a raw "
        "dataset and a profile with two to four figures (histograms, missing values, class "
        "balance or target spread, correlation).\n"
        "3. **Clean.** *You do:* approve or skip each proposed fix. *You get:* a cleaned "
        "dataset, a re-runnable `clean.py`, and the train/validation/test split.\n"
        "4. **Model.** *You do:* pick a model family or let the assistant recommend one. "
        "*You get:* a generated training project (`data.py`, `model.py`, `train.py`, "
        "`evaluate.py`, `config.json`).\n"
        "5. **Train.** *You do:* run the training and evaluation cells. *You get:* a live "
        "loss/metric chart, a saved checkpoint, and validation figures with captions.\n"
        "6. **Report.** *You do:* confirm you want the held-out test set scored. *You get:* "
        "`report.md` with the final numbers and every figure, written once per best model.\n\n"
        "**Two-click script cells.** A script cell starts as `%load train.py`: run it once to "
        "pull the code into the cell so you can read (and edit) it — this comments out the "
        "`%load` line and does nothing else. Run the same cell again to execute the code it "
        "now contains. When it finishes, go back to the assistant cell above it and run that "
        "again for the debrief.\n\n"
        "**Where your files are.** Everything the assistant writes lives on Drive under "
        "`MyDrive/ml_agent/projects/<your project name>/` — the raw and cleaned data, every "
        "script, `config.json`, the figures in `plots/`, and `report.md`. Nothing here is "
        "hidden from you; every generated file is plain, readable Python or JSON.\n\n"
        "**If you get stuck.** *Train again* (near the bottom) reruns training after you edit "
        "`config.json` by hand. *Ask about any term* looks up one word without rerunning "
        "anything. *Redo a stage* resets a stage and everything after it, then reruns from "
        "there. Click any highlighted term in the assistant's messages for an explanation; "
        "clicking does nothing while a cell is waiting for you to type an answer — click after "
        "the cell finishes, or run `colab.explain('term')` in its own cell."
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
        "#@markdown Tell the assistant about your project. Every box has a hint. If you are "
        "unsure, the default is a sensible start; you can change any answer later with the "
        "*Redo a stage* cell at the bottom.",
        "#@markdown **PROJECT_NAME** — A short name for this project's folder on Drive, e.g. "
        "`house_prices`. Letters, numbers and underscores.",
        "PROJECT_NAME = 'my_first_project'  #@param {type:'string'}",
        "#@markdown **LEARNING_LEVEL** — How much explanation you want. *Beginner* explains "
        "every term and step; *Intermediate* explains the reasoning; *Expert* shows only "
        "numbers and code.",
        "LEARNING_LEVEL = 'Intermediate - explain the key ideas'  #@param "
        "['Beginner - explain everything as we go', 'Intermediate - explain the key ideas', "
        "'Expert - just the numbers']",
        "#@markdown **GOAL** — One or two plain sentences on what you want to predict and "
        "why, e.g. `predict whether a customer will cancel next month so we can offer a "
        "discount`.",
        "GOAL = 'Predict which customers churn'  #@param {type:'string'}",
        "#@markdown **TASK** — *Tabular classification* predicts a category (yes/no, which "
        "type). *Tabular regression* predicts a number (a price, a temperature). If your "
        "answer is a label, choose classification.",
        "TASK = 'Tabular classification'  #@param ['Tabular classification', "
        "'Tabular regression']",
        "#@markdown **METRIC** — How success is scored. *Accuracy* is the share of correct "
        "predictions (fine when classes are balanced); *F1* is better when one class is rare; "
        "*RMSE* and *MAE* are the average size of the error for numbers, lower is better; "
        "*R2* is the fraction of the variation in the number your model explains, 1 is "
        "perfect. Unsure? Accuracy for classification, RMSE for regression. Only the "
        "metrics that fit your TASK are accepted.",
        "METRIC = 'accuracy'  #@param ['accuracy', 'f1', 'rmse', 'mae', 'r2']",
        "#@markdown **TARGET_VALUE** — The metric value you would be happy with. For accuracy "
        "or F1 a fraction like `0.9` (90%). For RMSE or MAE it is in the units of the thing "
        "you predict, e.g. `5000` for prices in pounds. Unsure? Keep `0.9` for "
        "classification; the assistant will discuss it.",
        "TARGET_VALUE = 0.9  #@param {type:'number'}",
        "#@markdown **DATA_SOURCE** — *Synthetic data* invents a realistic dataset so you can "
        "learn the process safely. *Upload or Google Drive path* uses your own CSV, Excel or "
        "Parquet file on Google Drive. *HuggingFace Hub dataset* searches a public library of "
        "datasets by keyword.",
        "DATA_SOURCE = 'Synthetic data'  #@param ['Synthetic data', "
        "'Upload or Google Drive path', 'HuggingFace Hub dataset']",
        "#@markdown **MINUTES_PER_RUN** — How long one training run may take. `10` is plenty "
        "for small tables; the assistant keeps its settings within this budget.",
        "MINUTES_PER_RUN = 10  #@param {type:'integer'}",
        "#@markdown **MAX_ROUNDS** — How many times the assistant may suggest an improvement "
        "and retrain after the first run. `3` to `5` is typical.",
        "MAX_ROUNDS = 5  #@param {type:'integer'}",
        "#@markdown **GPU** — Tables train on the CPU, so keep the no-GPU option for now. A "
        "GPU only matters for images, which come in a later milestone.",
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
        "#@markdown Get the data. Only the boxes for the data source you chose above matter; "
        "the others are ignored. When the assistant stops and names the next cell, run that "
        "cell.",
        "#@markdown **N_ROWS** — Synthetic only. How many example rows to invent. `1000` is "
        "enough to learn with and trains in seconds.",
        "N_ROWS = 1000  #@param {type:'integer'}",
        "#@markdown **N_FEATURES** — Synthetic only. How many input columns (things the model "
        "can look at). `8` is a good start.",
        "N_FEATURES = 8  #@param {type:'integer'}",
        "#@markdown **N_CLASSES** — Synthetic only, classification. How many categories the "
        "label can take; `2` for yes/no.",
        "N_CLASSES = 2  #@param {type:'integer'}",
        "#@markdown **CLASS_BALANCE** — Synthetic only. `0.5` means the classes are equally "
        "common; `0.9` makes one class rare, which is when accuracy misleads and F1 helps.",
        "CLASS_BALANCE = 0.5  #@param {type:'number'}",
        "#@markdown **NOISE** — Synthetic only. `0` makes the pattern easy to learn; `0.3` "
        "gives messy, realistic data.",
        "NOISE = 0.1  #@param {type:'number'}",
        "#@markdown **INJECT_QUIRKS** — Synthetic only. Adds real-world problems (missing "
        "values, duplicates, an ID column) so the cleaning step has something to fix. Keep it "
        "on while learning.",
        "INJECT_QUIRKS = True  #@param {type:'boolean'}",
        "#@markdown **DRIVE_PATH** — Drive only. The path to your file under MyDrive, e.g. "
        "`MyDrive/data/customers.csv`. Leave blank and the assistant lists the data files it "
        "can see so you can pick one; if it finds none it asks you to type the full path.",
        "DRIVE_PATH = ''  #@param {type:'string'}",
        "#@markdown **HF_QUERY** — HuggingFace only. A few keywords describing the dataset "
        "you want, e.g. `credit card fraud` or `iris`. Leave blank to be asked.",
        "HF_QUERY = ''  #@param {type:'string'}",
        "#@markdown **TARGET_COLUMN** — Drive and HuggingFace only. The column you want to "
        "predict. Leave blank: after loading the data the assistant shows your columns and "
        "asks you to pick, with its best guess marked.",
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
        "#@markdown Check and fix the data before training. The assistant audits the data "
        "and asks you to approve each fix, one at a time, in the output below. You do not "
        "need to know the fixes in advance.",
        "#@markdown **DROP_COLUMNS** — Optional. Columns you already know are useless for "
        "prediction (a name, a free-text note, an ID number), comma-separated. Leave blank: "
        "the audit finds ID and leaky columns itself and asks you about them.",
        "DROP_COLUMNS = ''  #@param {type:'string'}",
        "#@markdown **TRAIN_FRACTION** — Share of rows the model learns from. `0.7` (70%) is "
        "the usual choice; the rest is held back to check the model honestly.",
        "TRAIN_FRACTION = 0.7  #@param {type:'number'}",
        "#@markdown **VAL_FRACTION** — Share of rows used to check progress during training "
        "and tuning. `0.15` is usual. Whatever is left after training and validation becomes "
        "the test set, scored once at the very end.",
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
        "#@markdown Choose a model. The assistant explains the three options and recommends "
        "one for your data. If unsure, keep *Ask me after the explanation*.",
        "#@markdown **MODEL** — *Linear / logistic regression* is simple, fast and easy to "
        "read. *Random forest* is robust and copes with messy data. *Gradient boosting* is "
        "usually the most accurate on tables. *Ask me after the explanation* uses the "
        "assistant's recommendation.",
        "MODEL = 'Ask me after the explanation'  #@param "
        "['Ask me after the explanation', 'Linear / logistic regression', 'Random forest', "
        "'Gradient boosting']",
        "",
        "orch.run(until='train', answers={'codegen.model_type': MODEL})",
    ),
    script_cell("train", 0),
    script_cell("train", 1),
    code(
        "#@title 6. Report",
        "# Score the best model once on the test rows it has never seen, then write "
        "report.md. The assistant asks before touching the test set.",
        "orch.run(until='report')",
    ),
    script_cell("report", 0),
    code("orch.run()"),
    nbf.v4.new_markdown_cell(
        "## Train again\n\n"
        "For a manual experiment: edit `config.json` in the project folder yourself, then "
        "run the cell below. (The assistant's own guided tuning — where it proposes the "
        "config change for you — arrives in a later milestone.) The cell resets the train "
        "stage and re-prepares it, naming the `train.py` and `evaluate.py` cells above for "
        "you to run again. Run those two cells, then run the `orch.run()` cell above (just "
        "before this section) once more to log the new run and rewrite the report. "
        "`orch.waiting()` says which cells the assistant is still waiting on; "
        "`orch.debrief('train')` forces the debrief if Drive's timestamps lag."
    ),
    code("orch.reset('train')", "orch.run()"),
    nbf.v4.new_markdown_cell(
        "### Ask about any term\n\n"
        "Run this on its own, e.g. `colab.explain('overfitting')`, to look up one word "
        "without rerunning any stage."
    ),
    code("colab.explain('validation set')"),
    nbf.v4.new_markdown_cell(
        "### Redo a stage\n\n"
        "Use this when an earlier answer was wrong and you want that stage (and everything "
        "after it) redone from scratch: resets that stage and everything after it, then "
        "reruns."
    ),
    code("# orch.reset('intake'); orch.run()"),
]

# nbformat assigns each new cell a random id, which would make every rebuild of an
# otherwise-unchanged notebook look like a diff. Overwrite with the cell's position so
# two builds of the same `cells` list are byte-identical.
for _i, _cell in enumerate(cells):
    _cell["id"] = f"cell-{_i:02d}"

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
