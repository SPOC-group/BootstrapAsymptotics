function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Integer, ω::Real, V::Real, p::Real; rtol::Real)
    objective(z::Real) = abs2(z - ω) / (2V) + p * logistic_loss(y, z)
    gradient(_, z::Real) = (z - ω) / V + p * logistic_loss_der(y, z)
    hessian(_, z::Real) = inv(V) + p * logistic_loss_der2(y, z)

    scalarobj = NLSolvers.ScalarObjective(; f=objective, g=gradient, h=hessian)
    optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
    init = ω
    solver = NLSolvers.LineSearch(NLSolvers.Newton())
    options = NLSolvers.OptimizationOptions(; x_reltol=rtol, x_abstol=0.0)
    res = NLSolvers.solve(optprob, init, solver, options)

    prox = res.info.solution
    ∂ωprox = inv(1 + V * p * logistic_loss_der2(y, prox))  # implicit function theorem

    gₒᵤₜ = (prox - ω) / V
    ∂ωgₒᵤₜ = (∂ωprox - 1) / V

    return gₒᵤₜ, ∂ωgₒᵤₜ
end

function gₒᵤₜ_and_∂ωgₒᵤₜ(
    y::AbstractVector{<:Integer},
    ω::AbstractVector{<:Real},
    V::Diagonal{<:Real},
    p::AbstractVector{<:Real};
    rtol::Real,
)
    @assert length(y) == length(ω) == size(V, 1) == length(p) == 2
    gₒᵤₜ1, ∂ωgₒᵤₜ1 = gₒᵤₜ_and_∂ωgₒᵤₜ(y[1], ω[1], V[1, 1], p[1]; rtol)
    gₒᵤₜ2, ∂ωgₒᵤₜ2 = gₒᵤₜ_and_∂ωgₒᵤₜ(y[2], ω[2], V[2, 2], p[2]; rtol)
    gₒᵤₜ = SVector(gₒᵤₜ1, gₒᵤₜ2)
    ∂ωgₒᵤₜ = Diagonal(SVector(∂ωgₒᵤₜ1, ∂ωgₒᵤₜ2))
    return gₒᵤₜ, ∂ωgₒᵤₜ
end

function Z₀_and_∂μZ₀(
    y::AbstractVector{<:Integer}, μ::Real, v::Real, same_labels::Bool; rtol::Real
)
    function Z₀_and_∂μZ₀_integrand_same_labels(u::Real)
        z = u * sqrt(v) + μ
        σ = logistic(y[1] * z)
        σ_der = σ * (1 - σ)
        res = SVector(σ, y[1] * σ_der) * normpdf(u)
        return res
    end
    function Z₀_and_∂μZ₀_integrand_different_labels(u::Real)
        z = u * sqrt(v) + μ
        σ1 = logistic(y[1] * z)
        σ2 = logistic(y[2] * z)
        σ1_der = σ1 * (1 - σ1)
        σ2_der = σ2 * (1 - σ2)
        res = SVector(σ1 * σ2, y[1] * σ1_der * σ2 + y[2] * σ2_der * σ1) * normpdf(u)
        return res
    end
    bound = 10.0
    if same_labels
        @assert y[1] == y[2]
        double_integral, err = quadgk(
            Z₀_and_∂μZ₀_integrand_same_labels, -bound, bound; rtol
        )
    else
        double_integral, err = quadgk(
            Z₀_and_∂μZ₀_integrand_different_labels, -bound, bound; rtol
        )
    end
    Z₀ = double_integral[1]
    ∂μZ₀ = double_integral[2]
    return Z₀, ∂μZ₀
end

function update_hatoverlaps_summand(
    problem::Logistic,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::Overlaps{false},
    p::AbstractVector{<:Integer};
    rtol::Real,
)
    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    Q⁻¹ = inv(Q)
    Q_sqrt = sqrt(Q)
    v_star = ρ - dot(m, Q⁻¹ * m)

    Δm_hat, ΔQ_hat, ΔV_hat = zero(m), zero(Q), zero(V)

    for y1 in (-1, 1), y2 in (-1, 1)
        if same_labels(algo1, algo2) && y1 != y2
            continue
        end
        y = SVector(y1, y2)

        function integrand(u::AbstractVector)
            ω = Q_sqrt * u
            μ = dot(m, Q⁻¹ * ω)

            Z₀, ∂μZ₀ = Z₀_and_∂μZ₀(y, μ, v_star, same_labels(algo1, algo2); rtol)
            gₒᵤₜ, ∂ωgₒᵤₜ = gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, p; rtol)

            Im = ∂μZ₀ * gₒᵤₜ
            IQ = Z₀ * gₒᵤₜ * gₒᵤₜ'
            IV = -Z₀ * ∂ωgₒᵤₜ

            return vcat(Im, vec(IQ), IV.diag) * prod(normpdf, u)
        end

        bound = SVector(10.0, 10.0)
        integral, err = hcubature(integrand, -bound, +bound; rtol)

        Δm_hat += SVector(integral[1], integral[2])
        ΔQ_hat += SMatrix{2,2}(integral[3], integral[4], integral[5], integral[6])
        ΔV_hat += Diagonal(SVector(integral[7], integral[8]))
    end

    return Overlaps{true}(Δm_hat, ΔQ_hat, ΔV_hat)
end
