"""
Functions for the Bayes-optimal estimator
"""

function update_overlaps_BayesOpt(problem::Problem, q_hat::Real)
    (; λ, ρ) = problem
    R = 1.0 / (λ + q_hat)
    q = ρ .* R * q_hat # m = q = ρ - v
    return q
end

function Z₀_and_∂μZ₀(y::Integer, μ::Real, v::Real; rtol::Real)
    # need the logistic model for the BayesOpt estimator
    function Z₀_and_∂μZ₀_integrand(u::Real)
        z = u * sqrt(v) + μ
        σ = logistic(y * z)
        σ_der = σ * (1 - σ)
        res = SVector(σ, y * σ_der) * normpdf(u)
        return res
    end

    bound = 10.0
    double_integral, err = quadgk(
        Z₀_and_∂μZ₀_integrand, -bound, bound; rtol
    )
    Z₀ = double_integral[1]
    ∂μZ₀ = double_integral[2]
    return Z₀, ∂μZ₀
end

function gₒᵤₜ_BayesOpt(y::Integer, ω::Real, V::Real; rtol::Real)
    Z₀, ∂μZ₀ = Z₀_and_∂μZ₀(y, ω, V; rtol = rtol)

    gₒᵤₜ   = ∂μZ₀ / Z₀
    return gₒᵤₜₜ
end

function update_hatoverlaps_BayesOpt(
    problem::Logistic,
    q::Real;
    rtol::Real,
)
    (; α, ρ) = problem

    v_star = ρ - q

    ΔQ_hat = 0.0

    for y in (-1, 1)
        function integrand(u::Real)
            ω = sqrt(q) * u
            μ = ω

            Z₀, ∂Z₀ = Z₀_and_∂μZ₀(y, μ, v_star; rtol)
            gₒᵤₜ = (∂Z₀ / Z₀)

            return Z₀ * gₒᵤₜ^2. * prod(normpdf, u)
        end
        
        bound = 10.0
        integral, err = quadgk(integrand, -bound, +bound; rtol)
        ΔQ_hat += α * integral
    end

    return ΔQ_hat
end

function state_evolution_BayesOpt(
    problem::Logistic;
    rtol=1e-4,
    max_iteration=100,
)
    (; λ, ρ) = problem
    @assert λ == (1.0 / ρ)
    q::Real     = 1.0
    q_hat::Real = 1.0

    converged, nb_iterations = false, max_iteration

    for iter in 1:max_iteration
        new_q = update_overlaps_BayesOpt(problem, q_hat)
        new_q_hat = update_hatoverlaps_BayesOpt(problem, q; rtol)
        if (
            close_enough(new_q, q; rtol) &&
            close_enough(new_q_hat, q_hat; rtol)
        )
            converged, nb_iterations = true, iter
            break
        else
            q, q_hat = new_q, new_q_hat
        end
    end

    stats = (; converged, nb_iterations)
    return (; q, q_hat, stats)
end

function state_evolution_BayesOpt(
    problem::Ridge;
    rtol=1e-4,
    max_iteration=100,
)
    (; λ, ρ) = problem
    @assert λ == 1.0 / ρ
    # here we can afford to run the state evolution for 2d problem and 
    # only return the diagonal term
    res = state_evolution(problem, FullResampling(), FullResampling(), rtol=rtol, max_iteration=max_iteration)
    q = res.overlaps.Q[1, 1]
    q_hat = res.hatoverlaps.Q[1, 1]
    return (; q,  q_hat)
end