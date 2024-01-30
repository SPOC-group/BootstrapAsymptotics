# BootstrapAsymptotics

## Getting started

Open a Julia console and run:

```julia
pkg> add https://github.com/SPOC-group/BootstrapAsymptotics

shell> cd BootstrapAsymptotics

pkg> activate .

pkg> instantiate
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

```julia
pkg> activate docs

julia> include("docs/plots.jl")
```

## Notations

- `n`: number of samples
- `d`: dimension
- `α`: sampling ratio `n/d`
- `Δ`: noise variance
- `λ`: regularization parameter
- `m`, `Q`, `V`: overlaps
