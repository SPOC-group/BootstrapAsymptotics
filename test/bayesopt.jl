using BootstrapAsymptotics
using LinearAlgebra
using StableRNGs
using Test

rng = StableRNG(0)

n = 2000
α = 200

algo = BayesOpt()

for problem in (Logistic(; α), Ridge(; α))
    @testset "$(typeof(problem))" begin
        (; X, y, w) = sample_all(rng, problem, n)
        w_est = fit(rng, problem, algo, X, y)
        @test norm(w_est - w) / norm(w) < 0.2
    end
end
