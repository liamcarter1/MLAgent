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

## Milestone 3 (Tabular training end-to-end)

Prerequisite: a project that has completed the Milestone 2 checklist (clean data and splits).

1. Run the start cell. The codegen stage lists `data.py`, `model.py`, `train.py`, `config.json`
   in the project folder on Drive and shows a config table with a rationale. Click one
   `[[term]]` and confirm an explanation appears.
2. Answer "y" to the configuration question. The train stage prints the CPU/no-cost-gate note,
   then a loss/metric figure redraws in the cell as epochs complete.
3. When training ends: `runs.jsonl` has one line; `plots/` contains `run1_training.png` and the
   validation evaluation figures; the debrief mentions the best epoch.
4. The report stage asks before touching the test set. Answer "y". `eval_test.json` and
   `report.md` appear in the project folder; open `report.md` in Drive and check the figures render.
5. Run the "Train again" cell. A second run is logged as run 2. Re-run the report stage: if run 2
   is not better than run 1, the report is rewritten with both runs' history without asking about
   the test set again (the test split is evaluated once per best model). If run 2 became the new
   best run, the stage asks again before evaluating it on the test set.
6. Disconnect and reconnect the runtime, rerun the setup and start cells: the orchestrator prints
   "Nothing to do: all stages complete."

### Known Colab quirk

Occasionally the first question box after a stage banner never appears in Colab: the cell shows
the bold stage line, a blank tall output area, and interrupting shows a `KeyboardInterrupt` inside
ipykernel's `_input_request`. This is caused by Colab's frontend, not `mlagent`. Fix: Runtime >
Restart session, rerun the install and setup cells, then run `start` again. Never rely on a
`time.sleep` before `input()` to work around this: in testing that made `input()` return
immediately with no box shown at all.

## Milestone 4 (Learning mode and notebook-first pipeline)

Run in a fresh Colab runtime, on a new project name, with `LEARNING_LEVEL` set to
"Beginner - explain everything as we go" unless an item says otherwise.

1. **The two-click script cell works.** After cell 4 (`2. Data`) the assistant stops and
   says which cell to run. Cell 5 shows `%load profile.py`; run it once — the cell body is
   replaced by the source of `profile.py` with `%load` commented out. Run it again: the
   profile prints, four figures appear inline, and `profile_raw.json` lands in the project
   folder on Drive. If `%load` misbehaves, change `SCRIPT_CELL_FORMATS["load"]` in
   `mlagent/colab.py` to the `%run` form and re-run `python scripts/build_notebook.py`.
2. **Form fields answer the fixed questions.** Fill cell 3's form (project name, level,
   goal, task, metric, target, source, minutes, rounds, GPU) and run it: no `input()` box
   appears for any of those, and `spec.json` on Drive contains exactly what you typed,
   including `learning_level`.
3. **A blank form field falls back cleanly.** Clear `GOAL` in cell 3, reset the project
   (`orch.reset('intake')`) and run it: a one-line note names the empty field and an
   `input()` box asks for the goal instead.
4. **The data debrief carries captions.** Re-run cell 4 after cell 5: the profile table,
   then each of the four figures with a caption beneath it, then the narrative. At beginner
   level the caption has a second sentence about *your* data; switch the project to expert
   level and confirm the extra sentence disappears.
5. **Cleaning happens in the user's cell.** Cell 6 asks about each proposed fix in the
   output area, then names cell 7. `%load clean.py` shows the operations and the embedded
   `STEPS` list. Run it: `data/clean/data.csv` appears, and the before/after missing-value
   figure is drawn in the cell. Edit one entry in `STEPS`, run the cell again, then re-run
   cell 6 — the debrief reflects the edited cleaning.
6. **Three models, one recommendation.** Cell 8 with `MODEL = "Ask me after the
   explanation"` prints the three model descriptions, names a recommendation with a reason
   about your dataset, and writes a `config.json` whose keys match that family only
   (no `learning_rate` for a random forest). Re-run with `MODEL = "Random forest"` after
   `orch.reset('codegen')` and confirm `config.json` changes accordingly.
7. **The code walkthrough is level-appropriate.** At beginner level cell 8 ends with each
   of `data.py`, `model.py`, `train.py`, `evaluate.py` broken into its sections with an
   explanation under each. At expert level it prints only the filenames and section titles.
8. **The training curve animates.** Cell 9 (`%load train.py`, run twice) redraws one
   loss/metric figure in place as epochs complete — the transcript above it is not wiped —
   and leaves `plots/training_curves.png` and `checkpoints/best.joblib` on Drive. Cell 10
   (`evaluate.py`) writes `eval_val.json` and the validation figures.
9. **Re-running the scripts logs a second run.** Edit `epochs` in `config.json`, then run
   the cell under "Train again" (`orch.reset('train')` then `orch.run()`): the message
   re-prepares the train stage and names cells 9 and 10 to run again. Run those two cells,
   then run cell 13 (`orch.run()`) once more: `runs.jsonl` gains run 2,
   `plots/run2_training.png` exists, and the debrief compares the two runs.
10. **The test set is touched once, on purpose.** Resetting the train stage also resets the
    report stage, so the same run of cell 13 from item 9 continues straight into the report
    stage's own re-prepare: it names whichever of run 1 / run 2 now has the better
    validation metric and asks before evaluating it on the test set. Answer "y", run cell 12
    (`%run evaluate.py --split test`), then run cell 13 again to finish. `eval_test.json`
    records `run_id` equal to that best run, and `report.md` opens on Drive with every
    figure rendering. Repeat items 9-10 once more with a config change that does not beat
    the existing best run: the report stage instead says the test set was already evaluated
    for the current best run and rewrites the report without touching it again.
11. **A runtime reset resumes at the right phase.** Runtime > Disconnect and delete
    runtime. Re-run cells 1-3 only, then run cell 11: the orchestrator picks up where it
    was (or reports nothing to do) without re-asking any earlier question. If a debrief
    insists the outputs are missing because Drive's timestamps lag, `orch.debrief('train')`
    forces it through.
12. **Upgrading an older project re-picks a model.** A project created before Milestone 4
    has a `config.json` with no `model_type` key. Opening it under this build, the codegen
    stage treats that as incomplete and re-runs, asking the model-family question again;
    this is expected, not a bug.
