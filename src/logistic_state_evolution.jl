function init_all_overlaps(
    problem::Logistic,
    algo1::Algorithm,
    algo2::Algorithm;
    rtol::Real,
    max_iteration::Integer,
    show_progress::Bool,
)
    (overlaps1, hatoverlaps1, stats1) = state_evolution(
        problem, algo1; rtol, max_iteration, show_progress
    )
    if algo1 != algo2
        (overlaps2, hatoverlaps2, stats2) = state_evolution(
            problem, algo2; rtol, max_iteration, show_progress
        )
    else
        overlaps2, hatoverlaps2 = deepcopy(overlaps1), deepcopy(hatoverlaps1)
    end

    m1, Q11, V11 = overlaps1.m, overlaps1.Q, overlaps1.V
    m2, Q22, V22 = overlaps2.m, overlaps2.Q, overlaps2.V
    m_hat1, Q_hat11, V_hat11 = hatoverlaps1.m, hatoverlaps1.Q, hatoverlaps1.V
    m_hat2, Q_hat22, V_hat22 = hatoverlaps2.m, hatoverlaps2.Q, hatoverlaps2.V

    m = SVector(m1, m2)
    Q = SMatrix{2,2}(Q11, m1 * m2, m1 * m2, Q22)
    V = Diagonal(SVector(V11, V22))

    m_hat = SVector(m_hat1, m_hat2)
    Q_hat = SMatrix{2,2}(Q_hat11, m_hat1 * m_hat2, m_hat1 * m_hat2, Q_hat22)
    V_hat = Diagonal(SVector(V_hat11, V_hat22))

    overlaps = Overlaps(m, Q, V)
    hatoverlaps = Overlaps(m_hat, Q_hat, V_hat)

    @assert problem.ρ >= dot(overlaps.m, inv(overlaps.Q) * overlaps.m)
    return (; overlaps, hatoverlaps)
end

function update_hatoverlaps(
    problem::Logistic,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::O,
    hatoverlaps::O;
    rtol::Real,
) where {O<:Overlaps{2}}
    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    Q⁻¹ = inv(Q)
    Q_sqrt = sqrt(Q)
    v_star = ρ - dot(m, Q⁻¹ * m)

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)

    for (p1, p2) in weight_range(algo1, algo2)
        p = SVector(p1, p2)
        proba = weight_dist(algo1, algo2, p1, p2) * size_mul(algo1, algo2)
        iszero(proba) && continue

        for y in discrete_labels(algo1, algo2)
            function integrand(u::AbstractVector)
                ω = Q_sqrt * u
                μ = dot(m, Q⁻¹ * ω)

                Z₀, ∂μZ₀ = Z₀_and_∂μZ₀(y, μ, v_star; rtol)
                gₒᵤₜ, ∂ωgₒᵤₜ = gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, p; rtol)

                Im = ∂μZ₀ * gₒᵤₜ
                IQ = Z₀ * gₒᵤₜ * gₒᵤₜ'
                IV = Z₀ * ∂ωgₒᵤₜ

                return vcat(Im, vec(IQ), vec(IV)) * prod(normpdf, u)
            end

            bound = SVector(10.0, 10.0)
            integral, err = hcubature(integrand, -bound, +bound; rtol)

            m_hat += α * proba * SVector(integral[1:2]...)
            Q_hat += α * proba * SMatrix{2,2}(integral[3:6]...)
            V_hat -= α * proba * Diagonal(SVector(integral[7], integral[10]))
        end
    end

    return O(hatoverlaps.m, Q_hat, hatoverlaps.V)
end
