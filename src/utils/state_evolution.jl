function state_evolution(
    problem::Problem, algos::Vararg{Algorithm,N}; rtol=1e-3, max_iteration=100
) where {N}
    (; overlaps, hatoverlaps) = init_all_overlaps(problem, algos...; rtol, max_iteration)

    converged, nb_iterations = false, max_iteration
    for iter in 1:max_iteration
        new_hatoverlaps = update_hatoverlaps(problem, algos..., overlaps, hatoverlaps; rtol)
        new_overlaps = update_overlaps(problem, algos..., overlaps, new_hatoverlaps; rtol)
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
