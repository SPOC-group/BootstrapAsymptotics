"""
Ensembling two estimators that are trained on the NLL fo the Gaussian distribution 
ℓ(y, z) = (y - μ̂)² / (2 * v̂) + 1/2 * log(v̂)
where v̂ is the variance between the two : (z[1] - z[2])² / 4

gradient w.r.t z = μ̂, v̂
noting α = (μ̂ - y) / v̂
∇ℓ = ((μ̂ - y) / v̂, - (μ̂ - y)² / ( 2 * v̂²)
   = (α, - α² / 2.0)
   

Adding the regularization part (z - ω) * V^{-1} * (z - ω) + cancelling the derivtive
"""

function loss_gaussian_ensemble(z::AbstractVector, y::Real, ω::AbstractVector, V_inv::AbstractMatrix; ε::Real = 1e-6)
    μ̂ = ( z[1] + z[2] ) / 2.0
    v̂ = (z[1] - z[2])^2 / 4.0 + ε
    return - (y - μ̂)^2. / (2 * v̂) - log(v̂) / 2.0 
end

function objective_gaussian_ensemble(z::AbstractVector, y::Real, ω::AbstractVector, V_inv::AbstractMatrix; ε::Real = 1e-6)
    return loss_gaussian_ensemble(z,y,ω,V_inv;ε=ε) + (z - ω)' * V_inv * (z - ω)
end


function gₒᵤₜ_and_∂ωgₒᵤₜ_gaussian_ensemble(y::Real, ω::AbstractVector, V::AbstractMatrix, V_inv::AbstractMatrix; rtol::Real, ε::Real = 1e-10)
    """
    NOTE : for overparametrized logistic we still use  the same loss (logistic loss)
    so we don't need to change this part of the code
    """
    objective(z::AbstractVector) = objective_gaussian_ensemble(z, y, ω, V_inv; ε = ε)
    
    gradient(_, z::AbstractVector) = ForwardDiff.gradient(objective)(z)
    hessian(_,  z::AbstractVector) = ForwardDiff.hessian(objective)(z)

    loss(z::AbstractVector) = loss_gaussian_ensemble(z, y, ω, V_inv; ε = ε)
    hessian_loss(z::AbstractVector)= ForwardDiff.hessian(loss)(z)

    scalarobj = NLSolvers.ScalarObjective(; f=objective, g=gradient, h=hessian)
    optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
    init = ω
    solver = NLSolvers.LineSearch(NLSolvers.Newton())
    options = NLSolvers.OptimizationOptions(; x_reltol=rtol, x_abstol=0.0)
    res = NLSolvers.solve(optprob, init, solver, options)

    prox = res.info.solution
    ∂ωprox = inv(1 + V * p * hessian_loss(prox))  # implicit function theorem

    gₒᵤₜ       = V_inv * (prox - ω)
    ∂ωgₒᵤₜ     = V_inv * (∂ωprox - 1)

    return gₒᵤₜ, ∂ωgₒᵤₜ
end

function Z₀_and_∂μZ₀(y::Real, μ::Real, v_star::Real, problem::EnsembledRidge)
    (; Δ) = problem
    return exp(- (y - μ)^2. / (2.0 * (v_star + Δ))) / sqrt(2.0 * pi * (v_star + Δ))
end


### for overparametrized logistic


function update_hatoverlaps_summand(
    problem::EnsembledRidge,
    algo1::NoResampling,
    algo2::NoResampling,
    overlaps::Overlaps{false},
    p::AbstractVector{<:Integer};
    rtol::Real,
)   

    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    Q⁻¹ = inv(Q)
    V⁻¹ = inv(V)
    Q_sqrt = sqrt(Q)
    v_star = ρ - dot(m, Q⁻¹ * m)

    Δm_hat, ΔQ_hat, ΔV_hat = zero(m), zero(Q), zero(V)

    # integrand on y and the student's local fields
    function integrand(u::AbstractVector)
        y = u[1]
        ω = Q_sqrt * u[2, 3]
        μ = dot(m, Q⁻¹ * ω)

        Z₀, ∂μZ₀ = Z₀_and_∂μZ₀(y, μ, v_star, problem)
        gₒᵤₜ, ∂ωgₒᵤₜ = gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, p; rtol)

        Im = ∂μZ₀ * gₒᵤₜ
        IQ = Z₀ * gₒᵤₜ * gₒᵤₜ'
        IV = -Z₀ * ∂ωgₒᵤₜ

        return vcat(Im, vec(IQ), IV.diag) * prod(normpdf, u[2, 3])
    end

    bound = SVector(10.0, 10.0, 10.0)
    integral, err = hcubature(integrand, -bound, +bound; rtol)

    Δm_hat += SVector(integral[1], integral[2])
    ΔQ_hat += SMatrix{2,2}(integral[3], integral[4], integral[5], integral[6])
    ΔV_hat += Diagonal(SVector(integral[7], integral[8]))

    return Overlaps{true}(Δm_hat, ΔQ_hat, ΔV_hat)
end
