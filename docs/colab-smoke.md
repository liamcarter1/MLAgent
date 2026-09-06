# Colab smoke checklist

Run after each milestone. Copy `mlagent/` to `MyDrive/ml_agent/mlagent/` first.

## Milestone 1 (Foundation)
- [ ] Open `notebooks/ML_Training_Agent.ipynb` in Colab, run install and setup cells without error.
- [ ] `colab.start('smoke1'); orch.run()` asks the intake questions inline; answer them.
- [ ] `MyDrive/ml_agent/projects/smoke1/spec.json` exists and matches the answers.
- [ ] The closing message shows highlighted terms; clicking one shows an explanation panel.
- [ ] `colab.explain('learning rate')` shows an explanation; `glossary.json` now contains it.
- [ ] Runtime > Disconnect and delete runtime. Rerun setup and start cells: `orch.run()` reports nothing to do (intake already complete).
- [ ] `orch.reset('intake'); orch.run()` reruns the interview.

## Milestone 2 (Tabular data + cleaning)
- [ ] Fresh project, intake with data source "Synthetic data": the data stage asks rows/features/classes/balance/noise/quirks and shows a profile table, histograms, missing-value matrix, class balance, correlation heatmap, and a narrative with clickable terms.
- [ ] The clean stage shows a report card, then asks to approve each fix (ID column, duplicates, missing values, messy categories, outliers). Approve all; the summary shows fewer rows/columns; `projects/<name>/clean.py`, `data/clean/data.csv`, `audit.json` exist on Drive.
- [ ] Split question: enter 0.9 and 0.3 and see the adjustment message.
- [ ] New project with data source "Upload or Google Drive path": upload a CSV to Drive, confirm it appears in the file list, pick it, pick the target column.
- [ ] New project with data source "HuggingFace Hub dataset": search "iris" (or any small tabular set), pick one, pick the target; the audit runs on it.
- [ ] Runtime reset after the data stage: `orch.run()` skips intake and data, resumes at clean.
- [ ] Click a term inside the report card; the explanation mentions the current dataset (target column or issue).
