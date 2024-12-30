"""
    Optimized MSE error = ρ - 2m + q₀ (so not by ensembling the models)
"""
function compute_optimal_λ(; sample_over_student_dim::Float64, Δ::Float64,ρ::Float64, κ1::Float64, κstar::Float64, student_over_teacher_dim::Float64 )::Float64
    function to_optimize(λ::AbstractArray)::Real
        if λ[1] < 0.0
            return Inf
        end
        problem = BootstrapAsymptotics.build_ridge_overparametrized(α = sample_over_student_dim, true_Δ = Δ, λ = λ[1], true_ρ = ρ, κ1 = κ1, κstar = κstar, student_over_teacher_dim = student_over_teacher_dim, Δ̂ = 1.0, base_optimal = false)
        result = BootstrapAsymptotics.state_evolution(problem, BootstrapAsymptotics.NoResampling(), BootstrapAsymptotics.NoResampling())
        return ρ - 2.0 *  result.overlaps.m[1] + result.overlaps.Q[1, 1]
    end

    function to_optimize(λ::Real)::Real
        return 0.0
    end

    λ0 = [1.0]
    res = Optim.optimize(to_optimize, λ0)
    return Optim.minimizer(res)[1]
end