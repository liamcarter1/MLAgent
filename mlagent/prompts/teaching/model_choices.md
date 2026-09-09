### Three kinds of model

<!--level:beginner,intermediate-->
A model is a recipe for turning the columns of a row into a prediction. These three
recipes differ in how much shape they can learn and how much data they need to learn it.
<!--/level-->

#### Linear / logistic regression

Fits one weight per column and adds them up. Fast, hard to break, and the weights are
readable: a positive weight means "more of this column, higher prediction".

<!--level:beginner,intermediate-->
It can only draw a straight line (or a flat plane), so it misses interactions like "high
income *and* short tenure". It is the right first choice on small datasets, because with
few rows a flexible model mostly learns the noise — that is [[overfitting]].
<!--/level-->

#### Random forest

Grows many decision trees on different random slices of the data and averages them. Each
tree is a chain of yes/no questions, so it handles interactions and needs no scaling.

<!--level:beginner,intermediate-->
Averaging many noisy trees cancels out their individual mistakes, which is why a forest
is hard to overfit badly. It is slower to predict than a linear model and the individual
trees are no longer readable once you have hundreds of them.
<!--/level-->

#### Gradient boosting

Also builds trees, but one at a time, each one trained to fix the errors the previous
ones made. Usually the strongest option on tabular data.

<!--level:beginner,intermediate-->
Because every tree chases the remaining error, boosting *can* overfit if you let it run
too long — which is exactly what the [[validation]] curve is for. The [[learning rate]]
controls how much of each correction is applied: lower is slower but steadier.
<!--/level-->
