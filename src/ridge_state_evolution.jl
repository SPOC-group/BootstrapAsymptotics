function update_hatoverlaps(
    problem::Ridge,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::O,
    hatoverlaps::O;
    rtol::Real,
) where {O<:Overlaps{2}}
    (; m, Q, V) = overlaps
    (; α, Δ, ρ) = problem

    Q⁻¹ = inv(Q)
    v_star = ρ - sum(m' * Q⁻¹ * m)
    B = vcat(m', m') * Q⁻¹ - I

    m_hat, Q_hat, V_hat = zero(m), zero(Q), zero(V)

    for (p1, p2) in weight_range(algo1, algo2)
        p = SVector(p1, p2)
        proba = weight_dist(algo1, algo2, p1, p2) * size_mul(algo1, algo2)
        iszero(proba) && continue

        P = Diagonal(p)
        G = inv(I + P * V) * P

        m_hat += (α * proba) * (G * SVector(1, 1))
        if algo1 isa LabelResampling && algo2 isa LabelResampling
            # TODO: what about ResidualBootstrap?
            Q_hat += (α * proba) * (G * (v_star .+ Δ * I(2) .+ B * Q * B') * G')
        else
            Q_hat += (α * proba) * (G * (v_star .+ Δ .+ B * Q * B') * G')
        end
        V_hat += (α * proba) * G
    end

    return O(m_hat, Q_hat, V_hat)
end
