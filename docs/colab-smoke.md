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
