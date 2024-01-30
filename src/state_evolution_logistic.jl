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

function Z₀_and_∂μZ₀(y::Integer, μ::Real, v::Real; rtol::Real)
    function Z₀_and_∂μZ₀_integrand(u::Real)
        z = u * sqrt(v) + μ
        σ = logistic(y * z)
        res = SVector(σ, y * σ * (1 - σ)) * normpdf(u)
        return res
    end
    bound = 10.0
    double_integral, err = quadgk(Z₀_and_∂μZ₀_integrand, -bound, bound; rtol)
    Z₀ = double_integral[1]
    ∂μZ₀ = double_integral[2]
    return Z₀, ∂μZ₀
end

function Z₀_and_∂μZ₀(
    y::AbstractVector{<:Integer}, μ::Real, v::Real, same_labels::Bool; rtol::Real
)
    if same_labels
        Z₀, ∂μZ₀ = Z₀_and_∂μZ₀(y[1], μ, v; rtol)
    else
        Z₀1, ∂μZ₀1 = Z₀_and_∂μZ₀(y[1], μ, v; rtol)
        Z₀2, ∂μZ₀2 = Z₀_and_∂μZ₀(y[2], μ, v; rtol)
        Z₀, ∂μZ₀ = Z₀1 * Z₀2, ∂μZ₀1 * ∂μZ₀2
    end
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
