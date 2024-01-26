using Pkg
Pkg.activate(@__DIR__)

using BootstrapAsymptotics
using RidgeBootstrapStateEvolution: RidgeBootstrapStateEvolution
using LogisticBootstrapStateEvolution: LogisticBootstrapStateEvolution
using LinearAlgebra
using Test

@time res1 = state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0),
    PairBootstrap(),
    PairBootstrap();
    rtol=1e-4,
    max_iteration=100,
)
@time res1_ref = RidgeBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(
    1.0, 1e-4, 1.0; relative_tolerance=1e-4, max_iteration=100
)

@test res1.overlaps.m ≈ res1_ref["m"] rtol = 1e-3
@test res1.overlaps.Q ≈ res1_ref["q"] rtol = 1e-3
@test res1.overlaps.V ≈ Diagonal(res1_ref["v"]) rtol = 1e-3

@test res1.hatoverlaps.m ≈ res1_ref["mhat"] rtol = 1e-3
@test res1.hatoverlaps.Q ≈ res1_ref["qhat"] rtol = 1e-3
@test res1.hatoverlaps.V ≈ Diagonal(res1_ref["vhat"]) rtol = 1e-3
