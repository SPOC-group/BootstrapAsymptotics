using BootstrapAsymptotics
using Statistics
using StableRNGs
using Test

rng = StableRNG(0)

for problem in [Ridge(), Logistic()]
    @testset "$(typeof(problem))" begin
        for algo in [
            PairBootstrap(; p_max=5), FullResampling(), Subsampling(0.8), LabelResampling()
        ]
            @testset "$(typeof(algo))" begin
                _, var_emp = bias_variance_empirical(rng, problem, algo; n=100, K=50)
                var_se = variance_state_evolution(
                    problem, algo; check_convergence=false, show_progress=false
                )
                @test var_emp ≈ var_se rtol = 2e-1
            end
        end
    end
end
