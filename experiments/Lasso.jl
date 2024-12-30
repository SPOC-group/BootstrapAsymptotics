using BootstrapAsymptotics
using Plots
using StableRNGs: StableRNG

rng = StableRNG(0)
λ = 1.0
d = 500

α_range = 0.1:0.05:2.0

# store the overlaps m and q
m = []
q = []

m_erm = []
q_erm = []

for α in α_range
    problem = BootstrapAsymptotics.LassoWithLassoTeacher(α = α,  λ = λ)

    result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)

    push!(m, result.overlaps.m[1])
    push!(q, result.overlaps.Q[1, 1])
    (;X, w, y) = sample_all(rng, problem; teacher_dim = d)

    ŵ = BootstrapAsymptotics.fit(problem, ERM(), X, y)

    push!(m_erm, ŵ' * w / d)
    push!(q_erm, ŵ' * ŵ / d)
end

plt = plot(α_range, q, label = "SE m")
scatter!(plt, α_range, q_erm, label = "Empirical m")

## just some tests

α = 0.01
(;X, w, y) = sample_all(rng, BootstrapAsymptotics.LassoWithLassoTeacher(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0); teacher_dim = 10000)
stephist(w, density = true)

(;X, w, y) = sample_all(rng, BootstrapAsymptotics.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0); teacher_dim = 10000)
stephist!(w, density = true)