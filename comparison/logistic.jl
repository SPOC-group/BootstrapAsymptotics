using Pkg
Pkg.activate(@__DIR__)

using BootstrapAsymptotics
using RidgeBootstrapStateEvolution: RidgeBootstrapStateEvolution
using LogisticBootstrapStateEvolution: LogisticBootstrapStateEvolution
using LinearAlgebra
using Test

@time res1 = state_evolution(
    Logistic(; α=2.0, λ=1.0),
    PairBootstrap(; p_max=10),
    PairBootstrap(; p_max=10);
    rtol=1e-4,
    max_iteration=100,
)
@time res1_ref = LogisticBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(
    2.0, 1.0; max_weight=10, reltol=1e-4, max_iteration=100
)

@time res2 = state_evolution(
    Logistic(; α=2.0, λ=1.0),
    PairBootstrap(; p_max=10),
    FullResampling();
    rtol=1e-4,
    max_iteration=100,
)
@time res2_ref = LogisticBootstrapStateEvolution.state_evolution_bootstrap_full(
    2.0, 1.0; max_weight=10, reltol=1e-4
)

for (res, res_ref) in zip((res1, res2), (res1_ref, res2_ref))
    @test res.overlaps.m ≈ res_ref["m"] rtol = 1e-2
    @test res.overlaps.Q ≈ res_ref["q"] rtol = 1e-2
    @test res.overlaps.V ≈ Diagonal(res_ref["v"]) rtol = 1e-2

    @test res.hatoverlaps.m ≈ res_ref["mhat"] rtol = 1e-2
    @test res.hatoverlaps.Q ≈ res_ref["qhat"] rtol = 1e-2
    @test res.hatoverlaps.V ≈ Diagonal(res_ref["vhat"]) rtol = 1e-2
end
