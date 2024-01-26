function update_overlaps(
    problem::Logistic,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::O,
    hatoverlaps::O;
    rtol::Real,
) where {O<:Overlaps{2}}
    m_hat, Q_hat, V_hat = hatoverlaps.m, hatoverlaps.Q, hatoverlaps.V
    (; λ) = problem
    R = inv(λ * I + V_hat)
    Q = R * (m_hat * m_hat' + Q_hat) * R'
    return O(overlaps.m, Q, overlaps.V)
end

## Update hat overlaps

coherent_labels(::Algorithm, ::Algorithm, y1::Integer, y2::Integer) = y1 == y2
coherent_labels(::LabelResampling, ::LabelResampling, y1::Integer, y2::Integer) = true

function update_hatoverlaps(
    problem::Logistic,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::O,
    hatoverlaps::O;
    rtol::Real,
) where {O<:Overlaps{2}}
    @info "Update 2d"
    (; m, Q, V) = overlaps
    (; α, ρ) = problem

    Q⁻¹ = inv(Q)
    Q_sqrt = sqrt(Q)
    v_star = ρ - dot(m, Q⁻¹ * m)

    Q_hat11, Q_hat22 = hatoverlaps.Q[1, 1], hatoverlaps.Q[2, 2]

    Q_hat = zero(Q)

    for p1 in weight_range(algo1), p2 in weight_range(algo2)
        p = SVector(p1, p2)
        proba = weight_dist(algo1, algo2, p1, p2)
        iszero(proba) && continue

        for y1 in (-1, 1), y2 in (-1, 1)
            coherent_labels(algo1, algo2, y1, y2) || continue
            y = SVector(y1, y2)

            function integrand(u::AbstractVector)
                ω = Q_sqrt * SVector(u[1], u[2])
                μ = dot(m, Q⁻¹ * ω)
                z = v_star * u[3] + μ

                if algo1 isa LabelResampling && algo2 isa LabelResampling
                    Z₀ = logistic(y[1] * z) * logistic(y[2] * z)
                else
                    Z₀ = logistic(y[1] * z)
                end
                gₒᵤₜ1 = first(gₒᵤₜ_and_∂ωgₒᵤₜ(y[1], ω[1], V[1, 1], p[1]; rtol))
                gₒᵤₜ2 = first(gₒᵤₜ_and_∂ωgₒᵤₜ(y[2], ω[2], V[2, 2], p[2]; rtol))
                return Z₀ * gₒᵤₜ1 * gₒᵤₜ2 * prod(normpdf, u)
            end

            bound = SVector(7.0, 7.0, 7.0)
            Q_hat_offdiag, err = hcubature(integrand, -bound, +bound; rtol, maxevals=100)
            Q_hat +=
                α * proba * SMatrix{2,2}(Q_hat11, Q_hat_offdiag, Q_hat_offdiag, Q_hat22)
        end
    end

    return O(hatoverlaps.m, Q_hat, hatoverlaps.V)
end

## State evolution

function init_all_overlaps(
    problem::Logistic, algo1::Algorithm, algo2::Algorithm; rtol::Real, max_iteration
)
    @info "init"
    (overlaps1, hatoverlaps1, stats1) = state_evolution(problem, algo1; rtol, max_iteration)
    if algo1 != algo2
        (overlaps2, hatoverlaps2, stats2) = state_evolution(
            problem, algo2; rtol, max_iteration
        )
    else
        overlaps2, hatoverlaps2 = deepcopy(overlaps1), deepcopy(hatoverlaps1)
    end

    @info "off diag"
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
    return (; overlaps, hatoverlaps)
end
