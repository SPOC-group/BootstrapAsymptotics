function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Integer, ω::Real, V::Real, p::Real)
    objective(z::Real) = abs2(z - ω) / (2V) + p * logistic_loss(y, z)
    gradient(_, z::Real) = (z - ω) / V + p * logistic_loss_der(y, z)
    hessian(_, z::Real) = inv(V) + p * logistic_loss_der2(y, z)

    scalarobj = NLSolvers.ScalarObjective(; f=objective, g=gradient, h=hessian)
    optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
    init = ω

    # TODO : No control on the precision of the solution
    res = NLSolvers.solve(
        optprob,
        init,
        NLSolvers.LineSearch(NLSolvers.Newton()),
        NLSolvers.OptimizationOptions(; x_reltol=1e-4, x_abstol=0.0),
    )

    prox = res.info.solution
    ∂ωprox = inv(1 + V * p * logistic_loss_der2(y, prox))  # implicit function theorem

    gₒᵤₜ = (prox - ω) / V
    ∂ωgₒᵤₜ = (∂ωprox - 1) / V

    return gₒᵤₜ, ∂ωgₒᵤₜ
end

function Z₀_and_∂ωZ₀(y::Integer, μ::Real, v::Real)
    Z₀_integrand(z) = logistic(y * (z * sqrt(v) + μ)) * normpdf(z) * SVector(1, z)
    double_result = quadgk(Z₀_integrand, -Inf, Inf; rtol=1e-5)[1]
    Z₀ = double_result[1]
    ∂ωZ₀ = double_result[2]
    return Z₀, ∂ωZ₀
end

## Update overlaps

function update_overlaps(
    problem::Logistic, algo::Algorithm, hatoverlaps::O
) where {O<:Overlaps{1}}
    m_hat, Q_hat, V_hat = hatoverlaps.m, hatoverlaps.Q, hatoverlaps.V
    (; λ, ρ) = problem
    m = ρ * m_hat / (λ + V_hat)
    Q = (ρ * m_hat^2 + Q_hat) / (λ + V_hat)^2
    V = inv(λ + V_hat)
    return O(m, Q, V)
end

## Update hat overlaps

function update_hatoverlaps(
    problem::Logistic, algo::Algorithm, overlaps::O
) where {O<:Overlaps{1}}
    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    v_star = ρ - m^2 / Q

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)

    for p in weight_range(algo)
        proba = weight_dist(algo, p)
        for y in (-1, 1)
            function triple_integrand(u)
                ω = sqrt(Q) * u
                μ = m * u / Q

                Z₀, ∂ωZ₀ = Z₀_and_∂ωZ₀(y, μ, v_star)
                gₒᵤₜ, ∂ωgₒᵤₜ = gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, p)

                Im = ∂ωZ₀ * gₒᵤₜ
                IQ = Z₀ * gₒᵤₜ^2
                IV = Z₀ * ∂ωgₒᵤₜ
                return SVector(Im, IQ, IV) * normpdf(u)
            end

            triple_result = quadgk(triple_integrand, -Inf, Inf)[1]
            m_hat += α * proba * triple_result[1]
            Q_hat += α * proba * triple_result[2]
            V_hat += α * proba * triple_result[3]
        end
    end

    return O(m_hat, Q_hat, V_hat)
end
