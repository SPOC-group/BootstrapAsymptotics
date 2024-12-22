#=
Functions for the Bayes-optimal estimator
=#

"""
Note : it makes more sense to put the Bayes opt in the problem because 1) we don't have the λ to care about 
and 2) we can combine with resampling methods
"""
### for overparametrization
"""
TODO : Put this somewhere else
"""
function marcenko_pastur_integral(f::Function, γ::Float64)::Float64
    # Define λ_minus and λ_plus
    λ_minus = (1.0 - sqrt(γ))^2
    λ_plus = (1.0 + sqrt(γ))^2
    
    # Define the function to integrate
    to_integrate(x) = f(x) * sqrt((λ_plus - x) * (x - λ_minus)) / (2.0 * π * γ * x)
    
    # Perform the integration over the range (λ_minus, λ_plus)
    integral, _ = quadgk(to_integrate, λ_minus, λ_plus)
    
    # Adjust the integral for γ > 1.0
    if γ > 1.0
        return integral + (1.0 - 1.0 / γ) * f(0.0)
    end

    return integral
end

"""
Update of the overlaps when we have 2 BO on two i.i.d but different random features
"""
function update_overlaps(problem::BayesOptimalRidgeOverparametrized, hatoverlaps::Overlaps{true};)::Overlaps{false}
    κκ1 = problem.κ1^2
    κκstar = problem.κstar^2

    q̂0 = hatoverlaps.Q[1, 1]
    q̂1 = hatoverlaps.Q[2, 2]
    m̂  = hatoverlaps.m[1]
    v̂  = hatoverlaps.V[1, 1]

    function to_integrate(z::Real)
        return (κκ1 * z * problem.true_ρ / (κκ1 * z + κκstar)).^2 / (1.0 + q̂0 * (κκ1 * z * problem.true_ρ / (κκ1 * z + κκstar)))
    end 

    q₀ = q̂0 * problem.student_over_teacher_dim * marcenko_pastur_integral(to_integrate, problem.student_over_teacher_dim)
    # my intuition is that the formula for q₁ as a function of m is unchanged
    # this is only valid when you have two different i.i.d random features 

    q₁ = (1.0 + q̂1 / m̂^2) * q₀^2 # m = q₀ 
    v = problem.ρ - q₀

    mvec = SVector(q₀, q₀)
    Qmat = SMatrix{2,2}(q₀, q₁, q₁, q₀)
    Vmat = SMatrix{2,2}(v, 0, 0, v)

    return Overlaps{false}(mvec, Qmat, Vmat)
end