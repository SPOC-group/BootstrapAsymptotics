function update_overlaps(problem::Problem, hatoverlaps::Overlaps{true};)
    m_hat, Q_hat, V_hat = hatoverlaps.m, hatoverlaps.Q, hatoverlaps.V
    (; λ, ρ) = problem
    R = inv(λ * I + V_hat)
    m = ρ .* R * m_hat
    Q = (R * (ρ .* m_hat * m_hat' + Q_hat) * R')
    V = R
    return Overlaps{false}(m, Q, V)
end

function integrate_for_qvm(problem::RidgeOverparametrized, hatoverlaps::Overlaps{true};)
    (; λ, κ1, κstar) = problem
    q̂ = hatoverlaps.Q[1, 1]
    v̂ = hatoverlaps.V[1, 1]
    m̂ = hatoverlaps.m[1]

    # in Python code, gamma is the inverse of η
    η = problem.student_over_teacher_dim 
    ηp = (κ1 * (1 + sqrt(η)))^2
    ηm = (κ1 * (1 - sqrt(η)))^2
    den = λ + κstar^2 * v̂
    aux = sqrt(((ηp + κstar^2) * v̂ + λ) * ((ηm + κstar^2) * v̂ + λ))
    aux2 = sqrt(((ηp + κstar^2) * v̂ + λ) / ((ηm + κstar^2) * v̂ + λ))
    
    IV = ((κstar^2 * v̂ + λ) * ((ηp + ηm) * v̂ + 2 * λ) - 
          2 * κstar^2 * v̂^2 * sqrt(ηp * ηm) - 
          2 * λ * aux) / (4 * η * v̂^2 * (κstar^2 * v̂ + λ) * κ1^2)
    IV += max(0, 1 - 1.0 / η) * κstar^2 / (λ + v̂ * κstar^2)
    
    I1 = (ηp * v̂ * (-3 * den + aux) + 
          4 * den * (-den + aux) + 
          ηm * v̂ * (-2 * ηp * v̂ - 3 * den + aux)) / 
         (4 * η * v̂^3 * κ1^2 * aux)
    I2 = (ηp * v̂ + 
          ηm * v̂ * (1 - 2 * aux2) + 
          2 * den * (1 - aux2)) / 
         (4 * η * v̂^2 * aux * κ1^2)
    I3 = (2 * v̂ * ηp * ηm + 
          (ηp + ηm) * den - 
          2 * sqrt(ηp * ηm) * aux) / 
         (4 * η * den^2 * κ1^2 * aux)
    IQ = (q̂ + m̂^2) * I1 + (2 * q̂ + m̂^2) * κstar^2 * I2 + q̂ * κstar^2^2 * I3
    IQ += max(0, 1 - 1.0 / η) * q̂ * κstar^2^2 / den^2
    
    IM = ((ηm + ηp + 2 * κstar^2) * v̂ + 2 * λ - 2 * aux) / 
         (4 * η * v̂^2 * κ1^2)
    
    return IV, IQ, IM
end

function update_overlaps(problem::RidgeOverparametrized, hatoverlaps::Overlaps{true};)
    IV, IQ, IM = integrate_for_qvm(problem, hatoverlaps)
    m̂ = hatoverlaps.m[1]
    v = IV
    m = m̂ * IM * sqrt(problem.student_over_teacher_dim)
    q0 = IQ
    q̂1 = hatoverlaps.Q[1, 2]
    q1 = (m̂^2 + q̂1) * IM^2 * problem.student_over_teacher_dim

    mvec = SVector(m, m)
    Qmat = SMatrix{2,2}(q0, q1, q1, q0)
    Vmat = SMatrix{2,2}(v, 0, 0, v)

    return Overlaps{false}(mvec, Qmat, Vmat)
end

# 

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
    if problem isa RidgeOverparametrized
        m_hat *= sqrt(problem.student_over_teacher_dim)
    end
    return Overlaps{true}(m_hat, Q_hat, V_hat)
end

"""
$(SIGNATURES)

Peform state evolution on a `problem` for the couple `(algorithm1, algorithm2)`, by creating and then iteratively updating overlaps and hat overlaps.

# Keyword arguments

- `rtol`: relative tolerance used at every step of the procedure, especially to check overlap convergence
- `max_iteration`: maximum number of overlap updates
- `show_progress`: whether to display a progress bar
"""
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
    show_progress::Bool=false,
)
    res = state_evolution_BayesOpt(problem; rtol, max_iteration)
    (; ρ) = problem
    overlaps = Overlaps{false}(
        SVector(res.q, res.q),
        SMatrix{2,2}(res.q, nothing, nothing, res.q),
        SMatrix{2,2}(ρ - res.q, nothing, nothing, ρ - res.q),
    )

    hatoverlaps = Overlaps{true}(
        SVector(res.q_hat, res.q_hat),
        SMatrix{2,2}(res.q_hat, nothing, nothing, res.q_hat),
        SMatrix{2,2}(res.q_hat, nothing, nothing, res.q_hat),
    )

    return (; overlaps, hatoverlaps)
end
