

function Iplus_closed_form(b::Real, A::Real)::Real
    λ = 1.0 # change this when using non-BO regularization
    b = b - λ # change of var. : -1.0 because we integrage on negative x 
    return exp(b^2 / (2 * A)) * sqrt(π / (2 * A)) * erfc(- b / sqrt(2 * A))
end


function Iminus_closed_form(b::Real, A::Real)::Real
    λ = 1.0
    b = b + λ # we integrate on positive x
    return exp(b^2 / (2 * A)) * sqrt(π /  (2 * A)) * erfc( b / sqrt(2 * A))
end


function Z_a_laplace(b::Real, A::Real)::Real
    return Iplus_closed_form(b, A) + Iminus_closed_form(b, A)
end

function fa_laplace(b::Real, A::Real)::Real
    return ForwardDiff.derivative( b -> log(Z_a_laplace(b, A)), b)
end

function fv_laplace(b::Real, A::Real)::Real
    return ForwardDiff.derivative( b -> fa_laplace(b, A), b)
end

function update_overlaps(problem::BayesOptimalLasso, hatoverlaps::Overlaps{true}; rtol=1e-3)
    m̂ = hatoverlaps.m[1]
    q̂ = hatoverlaps.Q[1, 1]
    v̂ = hatoverlaps.V[1, 1]

    # 
    function integrand_m(ε::SVector{2})
        return fa_laplace(m̂ * ε[1] + sqrt(q̂) * ε[2], v̂) * ε[1] * normpdf(ε[2]) * pdf(Laplace(0.0, 1.0), ε[1])
    end

    # depends only on the sum of two i.i.d gaussians -> integrate on 1 RV
    function integrand_q(ε::SVector{2})
        return fa_laplace(m̂ * ε[1] + sqrt(q̂) * ε[2], v̂)^2. * normpdf(ε[2]) * pdf(Laplace(0.0, 1.0), ε[1])
    end

    function integrand_v(ε::SVector{2})
        return fv_laplace(m̂ * ε[1] + sqrt(q̂) * ε[2], v̂) * normpdf(ε[2]) * pdf(Laplace(0.0, 1.0), ε[1])
    end

    bound = 10.0
    integral_m, err = hcubature(
            integrand_m, (-bound, -bound), (bound, bound); rtol
    )
    integral_q, err = hcubature( integrand_q, (-bound, -bound), (bound, bound); rtol )
    integral_v, err = hcubature( integrand_v, (-bound, -bound), (bound, bound); rtol )
        
    mvec = SVector(integral_m, integral_m)
    Qmat = SMatrix{2,2}(integral_q, 0.0, 0.0, integral_q)
    Vmat = SMatrix{2,2}(integral_v, 0.0, 0.0, integral_v)

    return Overlaps{false}(mvec, Qmat, Vmat)
end