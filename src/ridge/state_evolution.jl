## Update overlaps

function update_overlaps(
    problem::Ridge, algo1::Algorithm, algo2::Algorithm, hatoverlaps::O
) where {O<:Overlaps{2}}
    m_hat, Q_hat, V_hat = hatoverlaps.m, hatoverlaps.Q, hatoverlaps.V
    (; λ) = problem
    R = inv(λ * I + V_hat)
    m = R * m_hat
    Q = R * (m_hat * m_hat' + Q_hat) * R'
    V = R
    return O(m, Q, V)
end

## Update hat overlaps

function update_hatoverlaps(
    problem::Ridge, algo1::Algorithm, algo2::Algorithm, overlaps::O;
) where {O<:Overlaps{2}}
    (; m, Q, V) = overlaps
    (; α, Δ, ρ) = problem
    if algo1 isa FullResampling && algo2 isa FullResampling
        α *= 2
    end

    Q⁻¹ = inv(Q)
    v_star = ρ - sum(m' * Q⁻¹ * m)
    B = vcat(m', m') * Q⁻¹ - I

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)

    for p1 in weight_range(algo1), p2 in weight_range(algo2)
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

    return O(m_hat, Q_hat, V_hat)
end
