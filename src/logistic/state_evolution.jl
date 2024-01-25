function update_overlaps(
    problem::Logistic, algo1::Algorithm, algo2::Algorithm, hatoverlaps::O
) where {O<:Overlaps{2}}

end

## Update hat overlaps

function update_hatoverlaps(
    problem::Logistic, algo1::Algorithm, algo2::Algorithm, overlaps::O
) where {O<:Overlaps{2}} end

## State evolution

function state_evolution(
    problem::Problem, algo1::Algorithm, algo2::Algorithm; rtol=1e-4, max_iteration=100
)
    single_result = state_evolution(problem, algo1; rtol, max_iteration)
    single_overlaps, single_hatoverlaps = single_result.overlaps, single_result.hatoverlaps

    overlaps, hatoverlaps = init_overlaps(Val(2)), init_overlaps(Val(2))
    converged = false
    nb_iterations = max_iteration

    
    stats = (; converged, nb_iterations)
    return (; overlaps, hatoverlaps, stats)
end
