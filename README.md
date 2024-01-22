# BootstrapAsymptotics

## Setting

Gaussian input $X$, labels $y$ generated either with an additive Gaussian noise or with a logistic model.

## Algorithms

- Pair bootstrap : sample the dataset $X, y$ with replacement (or in our model with a Poisson law of parameter 1)
- Full resampling (of dataset $X, y$)
- Parametric residual bootstrap : fit the whole dataset $X, y$ by ERM, and generate new data $y$ using the ERM estimator 
as a new teacher and by estimating the noise variance with the residuals of the fit (i.e square error).
- Resampling of labels $y$

## State evolutions

|                                              | Ridge | Logistic |
| -------------------------------------------- | ----- | -------- |
| Pair bootstrap                               |       |          |
| full resampling                              |       |          |
| y resampling                                 |       |          |
| residual bootstrap                           |       |          |
| Correlation Pair bootstrap - Full resampling |       |          |

## Notations

- `n`: number of samples
- `d`: dimension
- `α`: sampling ratio `n/d`
- `Δ`: noise variance
- `λ`: regularization parameter
- `m`, `Q`, `V`: overlaps