using BootstrapAsymptotics
using JET
using LinearAlgebra
using Test

overlaps, hatoverlaps = state_evolution(
    Logistic(; α=1.0, λ=1e-2, ρ=1.0), PairBootstrap(; p_max=3); rtol=1e-2, max_iteration=10
);
