using BootstrapAsymptotics
using JET
using LinearAlgebra
using Test

## Running without errors or type instabilities

algo_pairs = []
for algo1 in [PairBootstrap(), SubsamplingBootstrap(), FullResampling()]
    push!(algo_pairs, (algo1, algo1))
    if algo1 != FullResampling()
        push!(algo_pairs, (algo1, FullResampling()))
    end
end
push!(algo_pairs, (LabelResampling(), LabelResampling()))

for problem in (Ridge(), Logistic())
    @testset "$(typeof(problem))" begin
        for (algo1, algo2) in algo_pairs
            @testset "$(typeof(algo1)) - $(typeof(algo2))" begin
                state_evolution(problem, algo1, algo2; max_iteration=2)
                @test_opt target_modules = (BootstrapAsymptotics,) state_evolution(
                    problem, algo1, algo2; max_iteration=2
                )
            end
        end
    end
end
