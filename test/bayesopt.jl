using BootstrapAsymptotics
using Revise
using LinearAlgebra
using StableRNGs
using Test

rng = StableRNG(0)

n = 2000
α = 10
algo = BayesOpt()

problem = Ridge(; ρ = 1.0, λ = 1.0, α = α)
state_evolution(problem, algo, algo)