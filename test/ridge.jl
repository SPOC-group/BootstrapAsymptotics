using BootstrapAsymptotics
using LinearAlgebra
using Test

overlaps, hatoverlaps = state_evolution(
    Ridge(); weight_dist=indep_poisson, α=1.0, λ=1e-4, σ²=1.0
);

@test overlaps.m ≈ [0.631987, 0.631987] rtol = 1e-3
@test overlaps.Q ≈ [2.34849 1.1545; 1.1545 2.34849] rtol = 1e-3
@test overlaps.V ≈ Diagonal([3680.13, 3680.13]) rtol = 1e-3

@test hatoverlaps.m_hat ≈ [0.00017173, 0.00017173] rtol = 1e-3
@test hatoverlaps.Q_hat ≈ [1.43915e-7 5.57537e-8; 5.57537e-8 1.43915e-7] rtol = 1e-3
@test hatoverlaps.V_hat ≈ Diagonal([0.00017173, 0.00017173]) rtol = 1e-3
