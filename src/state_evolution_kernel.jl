"""
State evolution in the Kernel regime
"""

function update_hatoverlaps(
    problem::KernelRidgeOverparametrized,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::Overlaps{false};
    rtol::Real,
)
    """
    Main difference with other function is that we don't rescale by α
    """
    (; m, Q, V) = overlaps

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)
    for p1 in weight_range(algo1), p2 in weight_range(algo2)
        p = SVector(p1, p2)
        proba = weight_dist(algo1, algo2, p1, p2)

        if !iszero(proba)
            Δhatoverlaps = update_hatoverlaps_summand(
                problem, algo1, algo2, overlaps, p; rtol
            )
            m_hat += sqrt(problem.δ) * proba * Δhatoverlaps.m
            Q_hat += proba * Δhatoverlaps.Q
            V_hat += proba * Δhatoverlaps.V
        end
    end

    return Overlaps{true}(m_hat, Q_hat, V_hat)
end

function update_overlaps(problem::KernelRidgeOverparametrized, hatoverlaps::Overlaps{true};)
    (; λ, κ1, κstar, δ) = problem
    m̂vec = hatoverlaps.m
    Q̂vec = hatoverlaps.Q
    V̂vec = hatoverlaps.V

    m̂ = m̂vec[1]
    q̂0 = Q̂vec[1, 1]
    q̂1 = Q̂vec[1, 2]
    v̂ = V̂vec[1, 1]
    
    deno::Float64 = λ + δ * κ1^2 * v̂

    v = (λ * (κ1^2 + κstar^2) + δ^2 * κ1^2 * κstar^2 * v̂) / (λ * deno)
    m = (sqrt(δ) * m̂ * κ1^2) / deno
    q0 = δ * κ1^4 * (q̂0 + m̂^2) / (λ + δ * κ1^2 * v̂)^2
    q1 = δ * κ1^4 * (q̂1 + m̂^2) / (λ + δ * κ1^2 * v̂)^2

    mvec = SVector(m, m)
    Qmat = SMatrix{2,2}(q0, q1, q1, q0)
    Vmat = SMatrix{2,2}(v, 0, 0, v)

    return Overlaps{false}(mvec, Qmat, Vmat)
end


## 
function state_evolution(
    problem::KernelRidgeOverparametrized,
    algo1::Algorithm,
    algo2::Algorithm;
    rtol=1e-4,
    max_iteration=100,
    show_progress::Bool=false,
)
    # we redefine a new fumction because : 
    # we'll ignore the parameters α and student_over_teacher_dim from problem because of the kernel limit

    overlaps, hatoverlaps = Overlaps{false}(), Overlaps{true}()

    converged, nb_iterations = false, max_iteration
    p = Progress(max_iteration; desc="State evolution", enabled=show_progress)

    for iter in 1:max_iteration
        next!(p)
        new_overlaps = update_overlaps(problem, hatoverlaps)
        new_hatoverlaps = update_hatoverlaps(problem, algo1, algo2, new_overlaps; rtol)
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
