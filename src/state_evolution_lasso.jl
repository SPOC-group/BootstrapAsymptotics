function fa(b_::Real, A_::Real, λ::Real; ε::Real = 1e-5) # sigma = 1 / A > 0, r = b / A
    if abs(b_) < λ
        return 0.0
    elseif b_ > λ
        return (b_ - λ) / (A_ + ε)
    else
        return (b_ + λ) / (A_ + ε)
    end
end

function fv(b_::Real, A_::Real, λ::Real; ε::Real = 1e-5)
    if abs(b_) < λ
        return 0.0
    else
        return 1.0 / (A_ + ε)
   end
end

function update_hatoverlaps(problem::Lasso, ::NoResampling, ::NoResampling, overlaps::Overlaps{false}; rtol::Real)
    m = overlaps.m[1]
    q = overlaps.Q[1, 1]
    v = overlaps.V[1, 1]

    ratio = 1.0 / (problem.Δ̂ + v)
    m̂ = problem.α * ratio
    q̂ = problem.α * ratio^2 * (problem.ρ + problem.Δ + q - 2.0 * m)
    v̂ = problem.α * ratio

    return Overlaps{true}(SVector(m̂, m̂), SMatrix{2, 2}(q̂, 0, 0, q̂), SMatrix{2, 2}(v̂,0,0,v̂))
end

# this one is for Lasso on Gaussian teacher 
# so θ_* is Gaussian
function update_overlaps(problem::Lasso, hatoverlaps::Overlaps{true}; rtol=1e-3)
    m̂ = hatoverlaps.m[1]
    q̂ = hatoverlaps.Q[1, 1]
    v̂ = hatoverlaps.V[1, 1]

    # 
    function integrand_m(ε::SVector{2})
        return fa(m̂ * ε[1] + sqrt(q̂) * ε[2], v̂, problem.λ) * ε[1] * normpdf(ε[1]) * normpdf(ε[2]) 
    end

    # depends only on the sum of two i.i.d gaussians -> integrate on 1 RV
    function integrand_q(ε::Real)
        return fa(sqrt(q̂ + m̂^2) * ε, v̂, problem.λ)^2. * normpdf(ε)
    end

    function integrand_v(ε::Real)
        return fv(sqrt(q̂ + m̂^2) * ε, v̂, problem.λ) * normpdf(ε)
    end

    bound = 10.0
    integral_m, err = hcubature(
            integrand_m, (-bound, -bound), (bound, bound); rtol
    )
    integral_q, err = quadgk( integrand_q, -bound, bound; rtol )
    integral_v, err = quadgk( integrand_v, -bound, bound; rtol )
        
    mvec = SVector(integral_m, integral_m)
    Qmat = SMatrix{2,2}(integral_q, 0.0, 0.0, integral_q)
    Vmat = SMatrix{2,2}(integral_v, 0.0, 0.0, integral_v)

    return Overlaps{false}(mvec, Qmat, Vmat)
end

function state_evolution(problem::Lasso, algo1::NoResampling, algo2::NoResampling; rtol=1e-4,max_iteration=1000,show_progress::Bool=false,)
    """
        Here only consider one learner (we dont compute the cross term yet) so no resampling
    """
    overlaps, hatoverlaps = Overlaps{false}(), Overlaps{true}()
    converged, nb_iterations = false, max_iteration
    p = Progress(max_iteration; desc="State evolution", enabled=show_progress)

    for iter in 1:max_iteration
        println(iter)
        next!(p)
        new_hatoverlaps = update_hatoverlaps(problem, algo1, algo2, overlaps; rtol)
        new_overlaps = update_overlaps(problem, new_hatoverlaps)
        if (
            close_enough(new_overlaps, overlaps; rtol) &&
            close_enough(new_hatoverlaps, hatoverlaps; rtol)
        )
            converged, nb_iterations = true, iter
            break
        else
            overlaps, hatoverlaps = new_overlaps, new_hatoverlaps
        end
    end

    stats = (; converged, nb_iterations)
    return (; overlaps, hatoverlaps, stats)
end