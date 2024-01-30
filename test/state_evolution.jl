using BootstrapAsymptotics
using JET
using LinearAlgebra
using Test

algo_pairs = []
for algo1 in [PairBootstrap(), SubsamplingBootstrap(), FullResampling()]
    push!(algo_pairs, (algo1, algo1))
    if algo1 != FullResampling()
        push!(algo_pairs, (algo1, FullResampling()))
    end
end
push!(algo_pairs, (LabelResampling(), LabelResampling()))

for problem in (Ridge(), Logistic())
    for (algo1, algo2) in algo_pairs
        state_evolution(problem, algo1, algo2; max_iteration=2)
    end
end

## Type stability

@test_opt target_modules = (BootstrapAsymptotics,) state_evolution(
    Ridge(), PairBootstrap(), PairBootstrap(); max_iteration=2
)

@test_opt target_modules = (BootstrapAsymptotics,) state_evolution(
    Logistic(), PairBootstrap(), PairBootstrap(); max_iteration=2, show_progress=false
)
