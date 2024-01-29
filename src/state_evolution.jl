function update_overlaps(
    problem::Problem,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::O,
    hatoverlaps::O;
    rtol::Real,
) where {O<:Overlaps{2}}
    m_hat, Q_hat, V_hat = hatoverlaps.m, hatoverlaps.Q, hatoverlaps.V
    (; λ) = problem
    R = inv(λ * I + V_hat)
    m = R * m_hat
    Q = (R * (m_hat * m_hat' + Q_hat) * R')
    V = R
    return O(m, Q, V)
end

function state_evolution(
    problem::Problem,
    algos::Vararg{Algorithm,N};
    rtol=1e-4,
    max_iteration=100,
    show_progress::Bool=true,
) where {N}
    (; overlaps, hatoverlaps) = init_all_overlaps(
        problem, algos...; rtol, max_iteration, show_progress
    )

    converged, nb_iterations = false, max_iteration
    p = Progress(max_iteration; desc="$N-d state evolution", enabled=show_progress)
    for iter in 1:max_iteration
        next!(p)
        # maybe update the overlaps first from the hat overlaps so by construction we have a "good" matrix
        new_overlaps = update_overlaps(problem, algos..., overlaps, hatoverlaps; rtol)
        new_hatoverlaps = update_hatoverlaps(
            problem, algos..., new_overlaps, hatoverlaps; rtol
        )
        if close_enough(new_overlaps, overlaps; rtol) &&
            close_enough(new_hatoverlaps, hatoverlaps; rtol)
            converged, nb_iterations = true, iter
            break
        else
            overlaps, hatoverlaps = new_overlaps, new_hatoverlaps
        end
    end

    stats = (; converged, nb_iterations)
    return (; overlaps, hatoverlaps, stats)
end
