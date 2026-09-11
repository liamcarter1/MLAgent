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
        "The assistant leads you through seven steps. Steps 1-4 each have a form cell you "
        "fill in. Step 5 (train) is two script cells you run: `train.py`, then "
        "`evaluate.py`. Steps 6 (tune) and 7 (report) are cells you just run; tune sends you "
        "back to the two train cells once per round. Some steps also hand you a script cell "
        "to run:\n\n"
        "1. **Interview.** *You do:* answer the form below (project name, goal, task, metric, "
        "data source, budget). *You get:* a saved project specification, `spec.json`.\n"
        "2. **Data.** *You do:* give the path, search keywords, or the size of the "
        "synthetic dataset (the source itself is chosen in step 1). *You get:* a raw "
        "dataset and a profile with two to four figures. For a table those are "
        "histograms, missing values, class balance or target spread, and correlation; "
        "for images they are example thumbnails, class balance, pixel intensity and the "
        "mean image per class.\n"
        "3. **Clean.** *You do:* approve or skip each proposed fix. *You get:* a cleaned "
        "dataset, a re-runnable `clean.py`, and the train/validation/test split.\n"
        "4. **Model.** *You do:* pick a model family or let the assistant recommend one. "
        "*You get:* a generated training project (`data.py`, `model.py`, `train.py`, "
        "`evaluate.py`, `config.json`).\n"
        "5. **Train.** *You do:* run the training and evaluation cells. *You get:* a live "
        "loss/metric chart, a saved checkpoint, and validation figures with captions.\n"
        "6. **Tune.** *You do:* pick one of the assistant's proposed changes (or stop), rerun "
        "the two train cells, and run the tune cell again. *You get:* a diagnosis of the "
        "curves, a comparison chart of every run, and a new run logged with the change that "
        "made it. Repeat until you stop, hit your target, or use up MAX_ROUNDS.\n"
        "7. **Report.** *You do:* confirm you want the held-out test set scored. *You get:* "
        "`report.md` with the final numbers and every figure, written once per best model.\n\n"
        "**Two-click script cells.** A script cell starts as `%load train.py`: run it once to "
        "pull the code into the cell so you can read (and edit) it — this comments out the "
        "`%load` line and does nothing else. Run the same cell again to execute the code it "
        "now contains. When it finishes, go back to the assistant cell above it and run that "
        "again for the debrief.\n\n"
        "**Where your files are.** Everything the assistant writes lives on Drive under "
        "`MyDrive/ml_agent/projects/<your project name>/` — the raw and cleaned data "
        "(`data.csv` for a table, `data.npz` plus `manifest.csv` for images), every "
        "script, `config.json`, the figures in `plots/`, and `report.md`. Nothing here is "
        "hidden from you; every generated file is plain, readable Python or JSON.\n\n"
        "**If you get stuck.** *Train again* (near the bottom) reruns training after you edit "
        "`config.json` by hand; the tune step is the guided way to do the same thing. "
        "*Ask about any term* looks up one word without rerunning "
        "anything. *Redo a stage* resets a stage and everything after it, then reruns from "
        "there. Click any highlighted term in the assistant's messages for an explanation; "
        "clicking does nothing while a cell is waiting for you to type an answer — click after "
        "the cell finishes, or run `colab.explain('term')` in its own cell."
    ),
    code(
        "%pip -q install anthropic markdown pandas numpy scikit-learn matplotlib pyarrow "
        "openpyxl pillow huggingface_hub datasets torch torchvision"
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
        "#@markdown **TASK** — *Tabular classification* predicts a category from a table "
        "(yes/no, which type). *Tabular regression* predicts a number from a table (a "
        "price, a temperature). *Image classification* sorts pictures into classes. If "
        "your answer is a label, choose one of the classification options.",
        "TASK = 'Tabular classification'  #@param ['Tabular classification', "
        "'Tabular regression', 'Image classification']",
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
        "for small tables; the assistant keeps its settings within this budget, and the cost "
        "gate compares each run's estimate against this before it starts.",
        "MINUTES_PER_RUN = 10  #@param {type:'integer'}",
        "#@markdown **MAX_ROUNDS** — How many times the assistant may suggest an improvement "
        "and retrain after the first run. `3` to `5` is typical.",
        "MAX_ROUNDS = 5  #@param {type:'integer'}",
        "#@markdown **GPU** — Tables train on the CPU, so keep the no-GPU option for them. "
        "Images are much faster on a GPU: pick *T4 GPU* for image tasks, and set the "
        "runtime type to match (Runtime > Change runtime type > T4 GPU). *Pretrained "
        "ResNet-18* really wants one.",
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
        "#@markdown Get the data. Only the boxes for the task and data source you chose "
        "above matter; the others are ignored. When the assistant stops and names the next "
        "cell, run that cell.",
        "#@markdown **N_ROWS** — Synthetic only. How many example rows to invent. `1000` is "
        "enough to learn with and trains in seconds.",
        "N_ROWS = 1000  #@param {type:'integer'}",
        "#@markdown **N_FEATURES** — Synthetic only, ignored for image tasks. How many "
        "input columns (things the model can look at). `8` is a good start.",
        "N_FEATURES = 8  #@param {type:'integer'}",
        "#@markdown **N_CLASSES** — Synthetic only, classification. How many categories "
        "the label can take; `2` for yes/no, or for images, 2 to 5.",
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
        "#@markdown **TARGET_COLUMN** — Drive and HuggingFace only, ignored for image "
        "tasks. The column you want to predict. Leave blank: after loading the data the "
        "assistant shows your columns and asks you to pick, with its best guess marked.",
        "TARGET_COLUMN = ''  #@param {type:'string'}",
        "#@markdown **N_IMAGES** — Image tasks, synthetic only. How many shape pictures to "
        "invent. `300` trains in under a minute on the CPU.",
        "N_IMAGES = 300  #@param {type:'integer'}",
        "#@markdown **IMAGE_SIZE** — Image tasks only. How big each picture is made, in "
        "pixels. `32` is fastest and fine for shapes; `64` is the balanced default; `128` "
        "sees the most detail but wants the GPU runtime, especially with ResNet-18. As a "
        "practical ceiling, `128` with a few thousand images wants the GPU runtime and "
        "several GB of RAM.",
        "IMAGE_SIZE = '64'  #@param ['32', '64', '128']",
        "#@markdown **DRIVE_FOLDER** — Image tasks, Drive only. The folder holding your "
        "pictures, with one subfolder per class, e.g. `MyDrive/pets` containing "
        "`cats/` and `dogs/`. Files that are not readable images are reported, never fatal.",
        "DRIVE_FOLDER = ''  #@param {type:'string'}",
        "#@markdown **HF_DATASET** — Image tasks, HuggingFace only. The dataset id to "
        "download, e.g. `cifar10` or `beans`.",
        "HF_DATASET = ''  #@param {type:'string'}",
        "#@markdown **MAX_IMAGES** — Image tasks, Drive and HuggingFace only. Cap on how "
        "many pictures to use; `0` means all of them. `2000` (the default) keeps your "
        "first run quick and still keeps every class.",
        "MAX_IMAGES = 2000  #@param {type:'integer'}",
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
        "    'data.n_images': N_IMAGES,",
        "    'data.image_size': IMAGE_SIZE,",
        "    'data.drive_folder': DRIVE_FOLDER,",
        "    'data.hf_dataset': HF_DATASET,",
        "    'data.max_images': MAX_IMAGES,",
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
        "#@markdown Choose a model. The assistant explains the options for your kind of "
        "data and recommends one. If unsure, keep *Ask me after the explanation* — that "
        "works for tables and images alike.",
        "#@markdown **MODEL** — For tables: *Linear / logistic regression* is simple and "
        "easy to read, *Random forest* is robust, *Gradient boosting* is usually the most "
        "accurate. For images: *Tiny CNN* is a fast sanity check, *Small CNN* is the "
        "default, *Pretrained ResNet-18* is the most accurate but wants the GPU runtime.",
        "MODEL = 'Ask me after the explanation'  #@param "
        "['Ask me after the explanation', 'Linear / logistic regression', 'Random forest', "
        "'Gradient boosting', 'Tiny CNN', 'Small CNN', 'Pretrained ResNet-18']",
        "#@markdown **PRICE_PER_UNIT** — What one Colab compute unit costs you, so the "
        "estimate can show money. Colab Pro is $9.99 for 100 compute units, so `0.0999` "
        "per unit; check your plan in the Resources panel.",
        "PRICE_PER_UNIT = 0.0999  #@param {type:'number'}",
        "#@markdown **CURRENCY** — Shown next to the cost estimate, e.g. `$` or `GBP`.",
        "CURRENCY = '$'  #@param {type:'string'}",
        "",
        "orch.run(until='train', answers={",
        "    'codegen.model_type': MODEL,",
        "    'train.price_per_unit': PRICE_PER_UNIT,",
        "    'train.currency': CURRENCY,",
        "})",
    ),
    script_cell("train", 0),
    script_cell("train", 1),
    code(
        "#@title 6. Tune",
        "# The assistant reads the run history, diagnoses the curves and proposes one to",
        "# three changes. Pick one in the output below (or edit it, or stop), run the",
        "# train.py and evaluate.py cells above again, then run this cell again to see",
        "# whether it helped. It keeps going until you stop, the target is met, or",
        "# MAX_ROUNDS is used up.",
        "orch.run(until='tune')",
    ),
    code(
        "#@title 7. Report",
        "# Score the best model once on the test rows it has never seen, then write "
        "report.md. The assistant asks before touching the test set.",
        "orch.run(until='report')",
    ),
    script_cell("report", 0),
    code("orch.run()"),
    nbf.v4.new_markdown_cell(
        "## Train again\n\n"
        "For a manual experiment: edit `config.json` in the project folder yourself, then "
        "run the cell below. (The *6. Tune* cell is the guided alternative: the assistant "
        "proposes the change for you.) This cell resets the train stage, the tuning loop and "
        "the report, then re-prepares training, naming the `train.py` and `evaluate.py` cells "
        "above for you to run again. Run those two cells, then the *6. Tune* cell to log the "
        "new run and start a fresh tuning loop, or the `orch.run()` cell, which runs the tune "
        "step first (answer *Stop tuning and write the report* to skip it) and then the "
        "report. `orch.waiting()` says which cells the assistant is still waiting on; "
        "`orch.debrief('train')` forces the debrief if Drive's timestamps lag; "
        "`orch.reset('tune')` restarts only the tuning loop."
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
