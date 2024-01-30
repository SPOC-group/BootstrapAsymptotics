function update_hatoverlaps_summand(
    problem::Ridge,
    algo1::Algorithm,
    algo2::Algorithm,
    overlaps::Overlaps{false},
    p::AbstractVector{<:Integer};
    rtol::Real,
)
    (; m, Q, V) = overlaps
    (; Δ, ρ) = problem

    Q⁻¹ = inv(Q)
    v_star = ρ - dot(m, Q⁻¹ * m)
    B = vcat(m', m') * Q⁻¹ - I
    P = Diagonal(p)
    G = inv(I + P * V) * P

    Δm_hat = G * SVector(1, 1)
    if algo1 isa LabelResampling && algo2 isa LabelResampling
        ΔQ_hat = G * (v_star .+ Δ * I(2) .+ B * Q * B') * G'
    else
        ΔQ_hat = G * (v_star .+ Δ .+ B * Q * B') * G'
    end
    ΔV_hat = G

    return Overlaps{true}(Δm_hat, ΔQ_hat, ΔV_hat)
end
