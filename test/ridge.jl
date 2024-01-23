using BootstrapAsymptotics
using JET
using LinearAlgebra
using Test

overlaps, hatoverlaps = state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0, ρ=1.0),
    PairBootstrap(; p_max=8),
    PairBootstrap(; p_max=8);
    rtol=1e-4,
    max_iteration=100,
);

@test_opt target_modules = (BootstrapAsymptotics,) state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0, ρ=1.0),
    PairBootstrap(; p_max=2),
    PairBootstrap(; p_max=2);
    max_iteration=2,
)

allocs = @allocated state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0, ρ=1.0),
    PairBootstrap(; p_max=2),
    PairBootstrap(; p_max=2);
    max_iteration=2,
)
@test_broken allocs == 0

@test overlaps.m ≈ [0.631987, 0.631987] rtol = 1e-3
@test overlaps.Q ≈ [2.34849 1.1545; 1.1545 2.34849] rtol = 1e-3
@test overlaps.V ≈ Diagonal([3680.13, 3680.13]) rtol = 1e-3

@test hatoverlaps.m ≈ [0.00017173, 0.00017173] rtol = 1e-3
@test hatoverlaps.Q ≈ [1.43915e-7 5.57537e-8; 5.57537e-8 1.43915e-7] rtol = 1e-3
@test hatoverlaps.V ≈ Diagonal([0.00017173, 0.00017173]) rtol = 1e-3

# other combinations

state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0, ρ=1.0),
    PairBootstrap(; p_max=8),
    FullResampling();
    rtol=1e-4,
    max_iteration=100,
);

state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0, ρ=1.0),
    FullResampling(),
    FullResampling();
    rtol=1e-4,
    max_iteration=100,
);

state_evolution(
    Ridge(; α=1.0, λ=1e-4, Δ=1.0, ρ=1.0),
    LabelResampling(),
    LabelResampling();
    rtol=1e-4,
    max_iteration=100,
);
