# BootstrapAsymptotics

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SPOC-group.github.io/BootstrapAsymptotics/dev/)
[![Build Status](https://github.com/SPOC-group/BootstrapAsymptotics/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gdalle/HiddenMarkovModels.jl/actions/workflows/test.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Code for the paper

> _Analysis of Bootstrap and Subsampling in High-dimensional Regularized Regression_
> 
> Clarté et al. (2024), [arXiv:2402.13622](https://arxiv.org/abs/2402.13622)

## Getting started

Open a Julia console and run:

```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/SPOC-group/BootstrapAsymptotics")
```

## Using the package

```julia
julia> using BootstrapAsymptotics

julia> (; overlaps) = state_evolution(Ridge(λ=0.1), PairBootstrap(), PairBootstrap());

julia> Matrix(overlaps.Q)
2×2 Matrix{Float64}:
 1.35642   0.799808
 0.799808  1.35642
```

## Reproducing the plots

Open the Jupyter notebooks in the `experiments` folder, select the Julia environment defined in the same folder, and run.

## Computing the large $\alpha$ rates

Open the Mathematica notebook `experiments/large_alpha_rates.nb`

## Notations

- `n`: number of samples
- `d`: dimension
- `α`: sampling ratio `n/d`
- `Δ`: noise variance
- `λ`: regularization parameter
- `m`, `Q`, `V`: overlaps
