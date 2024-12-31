using BootstrapAsymptotics
using Distributions: Normal, quantile
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

### compare the test error at different values of λ

λ_range = 0.1:0.1:3.0
α_fixed = 0.5

m_λ = []
q_λ = []

for λ in λ_range
    problem = BootstrapAsymptotics.LassoWithLassoTeacher(α = α_fixed,  λ = λ)
    result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)

    push!(m_λ, result.overlaps.m[1])
    push!(q_λ, result.overlaps.Q[1, 1])
end

κ = 0.1
z = quantile(Normal(), 1 - κ / 2)   
z_neg = quantile(Normal(), κ / 2)

plot(λ_range, sqrt.(2.0 .- 2 * m_λ .+ q_λ .+ Δ) * (z - z_neg), label = "Interval size at $(1 - κ)")
# scatter the min error among all the λ

# RESULTS FOR FIGURE OF THE PAPER where α = 0.5
α = 0.5
Δ = 1.0
ρ_gaussian = 1.0
ρ_laplace  = 2.0


# ====== GAUSSIAN TEACHER BELOW ====== 

# Ridge : λ = 0.1
problem = BootstrapAsymptotics.Ridge(α = α,  λ = 0.1, ρ = ρ_gaussian)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
m, q = result.overlaps.m[1], result.overlaps.Q[1, 1]
interval_size = sqrt(ρ_gaussian - 2 * m + q + Δ) * (z - z_neg)
println("Interval size at $(1 - κ) for Ridge λ = 0.1: $interval_size")

# Ridge : λ = 1.0
problem = BootstrapAsymptotics.Ridge(α = α,  λ = 1.0, ρ = ρ_gaussian)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
m, q = result.overlaps.m[1], result.overlaps.Q[1, 1]
interval_size = sqrt(ρ_gaussian - 2 * m + q + Δ) * (z - z_neg)
println("Interval size at $(1 - κ) for Ridge λ = 1.0: $interval_size")

# LASSO : λ = 1.0
problem = BootstrapAsymptotics.Lasso(α = α,  λ = 1.0, ρ = ρ_gaussian)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
m, q = result.overlaps.m[1], result.overlaps.Q[1, 1]
interval_size = sqrt(ρ_gaussian - 2 * m + q + Δ) * (z - z_neg)
println("Interval size at $(1 - κ) for LASSO λ = 1.0: $interval_size")

# ====== LASSO TEACHER BELOW ====== 

# LASSO with Laplace teacher : λ = 0.1
problem = BootstrapAsymptotics.LassoWithLassoTeacher(α = α,  λ = 0.1, ρ = ρ_laplace)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
m, q = result.overlaps.m[1], result.overlaps.Q[1, 1]
interval_size = sqrt(ρ_laplace - 2 * m + q + Δ) * (z - z_neg)
println("Interval size at $(1 - κ) for LASSO with Lasso teacher λ = 0.1: $interval_size")

# LASSO with Laplace teacher : λ = 1.0
problem = BootstrapAsymptotics.LassoWithLassoTeacher(α = α,  λ = 1.0, ρ = ρ_laplace)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
m, q = result.overlaps.m[1], result.overlaps.Q[1, 1]
interval_size = sqrt(ρ_laplace - 2 * m + q + Δ) * (z - z_neg)
println("Interval size at $(1 - κ) for LASSO with Lasso teacher λ = 1.0: $interval_size")

# Ridge with Laplace teacher : λ = 1.0
problem = BootstrapAsymptotics.Ridge(α = α,  λ = 1.0, ρ = ρ_laplace)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
m, q = result.overlaps.m[1], result.overlaps.Q[1, 1]
interval_size = sqrt(ρ_laplace - 2 * m + q + Δ) * (z - z_neg)
println("Interval size at $(1 - κ) for Ridge with Lasso teacher λ = 1.0: $interval_size")