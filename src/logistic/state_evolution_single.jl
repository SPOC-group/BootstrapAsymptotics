function logistic_der(x::Real)
    s = logistic(x)
    return s * (1 - s)
end

logistic_loss(y, z) = log1pexp(-y * z)
logistic_loss_der(y, z) = -y * logistic(-y * z)
logistic_loss_der2(y, z) = y^2 * logistic_der(-y * z)

function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Integer, ω::Real, V::Real, p::Real; rtol::Real)
    objective(z::Real) = abs2(z - ω) / (2V) + p * logistic_loss(y, z)
    gradient(_, z::Real) = (z - ω) / V + p * logistic_loss_der(y, z)
    hessian(_, z::Real) = inv(V) + p * logistic_loss_der2(y, z)

    scalarobj = NLSolvers.ScalarObjective(; f=objective, g=gradient, h=hessian)
    optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
    init = ω
    solver = NLSolvers.LineSearch(NLSolvers.Newton())
    options = NLSolvers.OptimizationOptions(; x_reltol=rtol, x_abstol=0.1, maxiter=10)

    # TODO : No control on the precision of the solution
    res = NLSolvers.solve(optprob, init, solver, options)

    prox = res.info.solution
    ∂ωprox = inv(1 + V * p * logistic_loss_der2(y, prox))  # implicit function theorem

    gₒᵤₜ = (prox - ω) / V
    ∂ωgₒᵤₜ = (∂ωprox - 1) / V

    return gₒᵤₜ, ∂ωgₒᵤₜ
end

## Update overlaps

function update_overlaps(
    problem::Logistic, algo::Algorithm, overlaps::O, hatoverlaps::O; rtol::Real
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
    problem::Logistic, algo::Algorithm, overlaps::O, hatoverlaps::O; rtol::Real
) where {O<:Overlaps{1}}
    @info "Update 1d"
    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    Q⁻¹ = inv(Q)
    Q_sqrt = sqrt(Q)
    v_star = ρ - dot(m, Q⁻¹ * m)

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)

    for p in weight_range(algo)
        proba = weight_dist(algo, p)
        iszero(proba) && continue

        for y in (-1, 1)
            function triple_integrand(u::AbstractVector)
                ω = Q_sqrt * u[1]
                μ = m * Q⁻¹ * ω
                z = v_star * u[2] + μ
                ∂ωμ = m * Q⁻¹

                Z₀ = logistic(y * z)
                ∂ωZ₀ = y * ∂ωμ * logistic_der(y * z)
                gₒᵤₜ, ∂ωgₒᵤₜ = gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, p; rtol)

                Im = ∂ωZ₀ * gₒᵤₜ
                IQ = Z₀ * gₒᵤₜ^2
                IV = Z₀ * ∂ωgₒᵤₜ
                return SVector(Im, IQ, IV) * prod(normpdf, u)
            end

            bound = SVector(7.0, 7.0)
            triple_integral, err = hcubature(triple_integrand, -bound, bound; rtol, maxevals=100)
            @show triple_integral err
            m_hat += α * proba * triple_integral[1]
            Q_hat += α * proba * triple_integral[2]
            V_hat += α * proba * triple_integral[3]
        end
    end

    return O(m_hat, Q_hat, V_hat)
end
