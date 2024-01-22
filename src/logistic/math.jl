function prox(y::Real, ω::Real, v::Real, p::Real)
    objective(z::Real) = abs2(z - ω) / (2v) + p * log1p(exp(-y * z))
    gradient(g, z::Real) = derivative(objective, z)
    hessian(H, z::Real) = derivative(gradient, z)

    scalarobj = NLSolvers.ScalarObjective(; f=objective, g=gradient, h=hessian)
    optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
    init = ω

    # TODO : No control on the precision of the solution
    res = NLSolvers.solve(
        optprob,
        init,
        NLSolvers.LineSearch(NLSolvers.Newton()),
        NLSolvers.OptimizationOptions(),
    )

    return res.info.solution
end

gₒᵤₜ(y::Real, ω::Real, v::Real, p::Real) = (prox(y, ω, v, p) - ω) / v

function Z₀(y::Real, μ::Real, v::Real)
    f(x) = logistic(y * (x * sqrt(v) + μ)) * normpdf(x)
    return quadgk(f, -Inf, Inf)[1]
end

function ∂ωZ₀(y::Real, μ::Real, v::Real)
    f(z) = z * logistic(y * (z * sqrt(v) + μ)) * normpdf(z)
    result = quadgk(integrand, -Inf, Inf)[1]
    return result / sqrt(v)
end
