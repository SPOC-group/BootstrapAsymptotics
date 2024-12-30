using BootstrapAsymptotics
using StableRNGs: StableRNG

rng = StableRNG(0)
λ = 0.25
d = 1000

α_range = 0.1:0.05:2.0

# store the overlaps m and q
m = []
q = []

m_erm = []
q_erm = []

for α in α_range
    problem = Lasso(α = α, ρ = 1.0, λ = λ, Δ = 1.0, Δ̂ = 1.0)

    result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)

    push!(m, result.overlaps.m[1])
    push!(q, result.overlaps.Q[1, 1])
    (;X, w, y) = sample_all(rng, problem; teacher_dim = d)

    ŵ = BootstrapAsymptotics.fit(problem, ERM(), X, y)

    push!(m_erm, ŵ' * w / d)
    push!(q_erm, ŵ' * ŵ / d)
end

using Plots

plt = plot(α_range, q, label = "SE m")
scatter!(plt, α_range, q_erm, label = "Empirical m")
# sho the plot
