### Tuning: reading the curves, then turning one knob

<!--level:beginner,intermediate-->
Tuning means changing the settings in `config.json` and training again. The assistant
looks at the [[validation loss]] curve from the last run, names what it sees, and proposes
one change at a time so you can tell what each change did.
<!--/level-->

#### The four shapes a curve can take

- **Overfitting.** Training loss keeps falling, validation loss turns upward. The model is
  memorising rows. Make it more cautious: bigger leaves, stronger regularisation, a smaller
  [[learning rate]].
- **Underfitting.** Both losses are still falling when training stops. Give it more: more
  [[epoch]]s, a bigger step, more capacity.
- **Learning rate too high.** Validation loss jumps up and down. Each update overshoots;
  take smaller steps.
- **Plateau.** Both losses are flat. More of the same will not help; change the shape of
  the model (bigger trees, more columns per tree) or switch family.

<!--level:beginner-->
Losses are the model's own score for being wrong: lower is better, and the training loss is
always a little flattering because the model has seen those rows. The validation loss is the
honest one, which is why the diagnosis reads that curve.
<!--/level-->

#### What each family's knobs do

**Gradient boosting.** `learning_rate` is how much of each new tree's correction is applied
(smaller is steadier, needs more epochs); `iters_per_epoch` is trees added per epoch;
`max_leaf_nodes` and `max_depth` are how big a tree can grow (bigger learns finer detail and
overfits sooner); `min_samples_leaf` is how many rows a leaf must hold (bigger is more
cautious); `l2_regularization` shrinks leaf values (bigger is more cautious).

**Random forest.** `trees_per_epoch` is how many trees each epoch adds (more is smoother);
`max_depth` and `min_samples_leaf` limit each tree as above; `max_features` is the share of
columns each tree may look at (smaller makes trees disagree more, which usually helps).

**Linear / logistic regression.** `learning_rate` is the step size; `alpha` is the
[[regularisation]] strength (bigger keeps the weights small and fights overfitting).

<!--level:beginner,intermediate-->
`epochs` and `early_stopping_patience` are shared: an epoch is one round of learning, and
patience is how many epochs without improvement the run tolerates before stopping.
<!--/level-->
