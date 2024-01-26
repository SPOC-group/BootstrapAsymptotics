using BootstrapAsymptotics
using JET
using LinearAlgebra
using Test

@test_opt target_modules = (BootstrapAsymptotics,) state_evolution(
    Ridge(), PairBootstrap(), PairBootstrap(); max_iteration=2
)

@test_opt target_modules = (BootstrapAsymptotics,) state_evolution(
    Logistic(), PairBootstrap(), PairBootstrap(); max_iteration=2, show_progress=false
)
