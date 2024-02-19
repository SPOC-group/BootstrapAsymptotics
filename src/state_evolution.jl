function update_overlaps(problem::Problem, hatoverlaps::Overlaps{true};)
    m_hat, Q_hat, V_hat = hatoverlaps.m, hatoverlaps.Q, hatoverlaps.V
    (; λ, ρ) = problem
    R = inv(λ * I + V_hat)
    m = ρ .* R * m_hat
    Q = (R * (ρ .* m_hat * m_hat' + Q_hat) * R')
    V = R
    return Overlaps{false}(m, Q, V)
end

function update_hatoverlaps(
    problem::Problem,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::Overlaps{false};
    rtol::Real,
)
    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)
    for p1 in weight_range(algo1), p2 in weight_range(algo2)
        p = SVector(p1, p2)
        proba = weight_dist(algo1, algo2, p1, p2)

        if !iszero(proba)
            Δhatoverlaps = update_hatoverlaps_summand(
                problem, algo1, algo2, overlaps, p; rtol
            )
            m_hat += α * proba * Δhatoverlaps.m
            Q_hat += α * proba * Δhatoverlaps.Q
            V_hat += α * proba * Δhatoverlaps.V
        end
    end
    return Overlaps{true}(m_hat, Q_hat, V_hat)
end

function state_evolution(
    problem::Problem,
    algo1::Algorithm,
    algo2::Algorithm;
    rtol=1e-4,
    max_iteration=100,
    show_progress::Bool=false,
)
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

# LC : I add a function state_evolution for the Bayes-optimal estimator, just for convenience. I cant compute the off diagonal overlaps
# so I replace them by nothing, it's ok because for now we don't need them 

function state_evolution(
    problem::Problem,
    ::BayesOpt,
    ::BayesOpt;
    rtol=1e-4,
    max_iteration=100,
    show_progress::Bool=false)

    res = state_evolution_BayesOpt(problem; rtol, max_iteration)
    (; ρ) = problem
    overlaps = Overlaps{false}(
        SVector(res.q, res.q),
        SMatrix{2, 2}(res.q, nothing, nothing, res.q),
        SMatrix{2, 2}(ρ - res.q, nothing, nothing, ρ - res.q),
    )

    hatoverlaps = Overlaps{true}(
        SVector(res.q_hat, res.q_hat),
        SMatrix{2, 2}(res.q_hat, nothing, nothing, res.q_hat),
        SMatrix{2, 2}(res.q_hat, nothing, nothing, res.q_hat),
    )

    return (; overlaps, hatoverlaps)
end