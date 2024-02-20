"""
Contains the code to run the BayesOpt estimator for logistic regression
"""

include("state_evolution_logistic.jl")

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector)
    n = length(y)
    g = zeros(n)
    dg = zeros(n)

    for i in 1:n
        g[i], dg[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i], 1.0; rtol = 1e-3)
    end

    return g, dg
end

function prior(b::AbstractVector, A::AbstractVector, λ::Real, ρ::Real)
    Σ = 1. ./ A
    R = b ./ A

    return R ./ (λ .* Σ .+ 1.0), Σ ./ (λ .* Σ .+ 1.0)
end

function gamp(problem::Logistic, X::AbstractMatrix, y::AbstractVector; max_iter::Integer = 100, rtol::Real = 1e-3)
    (; ρ, λ) = problem
    n, d = size(X)
    X_squared = X .* X

    vhat = ones(d)
    xhat = zeros(d)
    g    = zeros(n)

    for iteration in 1:max_iter
        V = X_squared * vhat

        ω = X * xhat - V .* g
        g, dg = channel(y, ω, V)

        A = - X_squared' * dg
        b = A .* xhat + X' * g
        
        xhat_old = copy(xhat)

        xhat, vhat = prior(b, A, λ, ρ)

        if norm(xhat - xhat_old) / norm(xhat) < rtol
            break
        end
    end

    return (; xhat, vhat)
end