## Update overlaps

function update_overlaps(
    problem::Ridge, algo1::Algorithm, algo2::Algorithm, hatoverlaps::HatOverlaps
)
    (; m_hat, Q_hat, V_hat) = hatoverlaps
    (; λ) = problem
    R = inv(λ * I + V_hat)
    m = R * m_hat
    Q = R * (m_hat * m_hat' + Q_hat) * R'
    V = R
    return Overlaps(m, Q, V)
end

## Update hat overlaps

function update_hatoverlaps(
    problem::Ridge, algo1::Algorithm, algo2::Algorithm, overlaps::Overlaps;
)
    (; m, Q, V) = overlaps
    (; α, Δ, ρ) = problem
    if algo1 isa FullResampling && algo2 isa FullResampling
        α *= 2
    end

    Q⁻¹ = inv(Q)
    v_star = ρ - sum(m' * Q⁻¹ * m)
    B = vcat(m', m') * Q⁻¹ - I

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)
    p1_range, p2_range = weight_ranges(algo1, algo2)

    for p1 in p1_range, p2 in p2_range
        proba = weight_dist(algo1, algo2, p1, p2)
        iszero(proba) && continue

        P = Diagonal(SVector(p1, p2))
        G = inv(I + P * V) * P

        m_hat += (α * proba) * (G * SVector(1, 1))
        if algo1 isa LabelResampling && algo2 isa LabelResampling
            Q_hat += (α * proba) * (G * (v_star .+ Δ * I .+ B * Q * B') * G')
        else
            Q_hat += (α * proba) * (G * (v_star .+ Δ .+ B * Q * B') * G')
        end
        V_hat += (α * proba) * G
    end

    return HatOverlaps(m_hat, Q_hat, V_hat)
end

## State evolution

function state_evolution(
    problem::Ridge,
    algo1::Algorithm,
    algo2::Algorithm;
    relative_tolerance=1e-4,
    max_iteration=100,
)
    overlaps, hatoverlaps = init_overlaps()
    for _ in 0:max_iteration
        new_hatoverlaps = update_hatoverlaps(problem, algo1, algo2, overlaps)
        new_overlaps = update_overlaps(problem, algo1, algo2, new_hatoverlaps)
        if relative_difference(new_overlaps, overlaps) < relative_tolerance
            return new_overlaps, new_hatoverlaps
        else
            overlaps, hatoverlaps = new_overlaps, new_hatoverlaps
        end
    end
    @warn "State evolution did not converge after $max_iteration iterations"
    return (; overlaps, hatoverlaps)
end
