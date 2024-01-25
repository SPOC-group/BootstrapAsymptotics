function state_evolution(
    problem::Problem, algos::Vararg{Algorithm,N}; rtol=1e-4, max_iteration=100
) where {N}
    overlaps, hatoverlaps = init_overlaps(Val(N)), init_overlaps(Val(N))
    converged = false
    nb_iterations = max_iteration
    for iter in 1:max_iteration
        new_hatoverlaps = update_hatoverlaps(problem, algos..., overlaps)
        new_overlaps = update_overlaps(problem, algos..., new_hatoverlaps)
        if close_enough(new_overlaps, overlaps; rtol) &&
            close_enough(new_hatoverlaps, hatoverlaps; rtol)
            converged = true
            nb_iterations = iter
            break
        else
            overlaps, hatoverlaps = new_overlaps, new_hatoverlaps
        end
    end
    stats = (; converged, nb_iterations)
    return (; overlaps, hatoverlaps, stats)
end
