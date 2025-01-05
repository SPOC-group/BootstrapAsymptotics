using BootstrapAsymptotics
using Distributions: Normal, quantile
using Plots

κ = 0.1
z = quantile(Normal(), 1 - κ / 2)   
z_neg = quantile(Normal(), κ / 2)

# plot(λ_range, sqrt.(2.0 .- 2 * m_λ .+ q_λ .+ Δ) * (z - z_neg), label = "Interval size at $(1 - κ)")
# scatter the min error among all the λ

# RESULTS FOR FIGURE OF THE PAPER where α = 0.5
α = 0.5
Δ = 1.0
ρ_gaussian = 1.0
ρ_laplace  = 2.0


# ====== GAUSSIAN TEACHER BELOW ====== 

println("======= GAUSSIAN TEACHER =======")
# Bayes optimal 
problem = BootstrapAsymptotics.Ridge(α = α,  λ = 1.0, ρ = ρ_gaussian)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(), BootstrapAsymptotics.NoResampling(); show_progress = false)
v = result.overlaps.V[1]
interval_size = sqrt(v + 1.0) * (z - z_neg)
println("Interval size at $(1 - κ) for Ridge Bayes optimal : $interval_size")

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

println("====== LAPLACE TEACHER ===== ")
# ====== LASSO TEACHER BELOW ====== 

# Bayes optimal with Laplace teacher 
problem = BootstrapAsymptotics.BayesOptimalLasso(α = α, ρ = ρ_laplace)
result = state_evolution(problem, BootstrapAsymptotics.NoResampling(),BootstrapAsymptotics.NoResampling(); show_progress = false)
v = result.overlaps.V[1]
interval_size = sqrt(v + 1.0) * (z - z_neg)
println("Interval size at $(1 - κ) for Bayes optimal LASSO with Laplace: $interval_size")

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

