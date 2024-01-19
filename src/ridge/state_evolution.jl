## Update overlaps

function update_overlaps(hatoverlaps::HatOverlaps; λ::Real)
    (; m̂, Q̂, V̂) = hatoverlaps
    R = inv(λ * I + V̂)
    m = R * m̂
    Q = R * (m̂ * m̂' + Q̂) * R'
    V = R
    return Overlaps(m, Q, V)
end

## Update hat overlaps

function update_hatoverlaps(
    overlaps::Overlaps; weight_dist::Function, α::Real, σ²::Real, ρ::Real, pmax::Integer
)
    (; m, Q, V) = overlaps

    Q⁻¹ = inv(Q)
    vstar = ρ - sum(m' * Q⁻¹ * m)
    B = vcat(m', m') * Q⁻¹ - I

    m̂, Q̂, V̂ = zero(m), zero(Q), zero(V)

    for p1 in 0:pmax, p2 in 0:pmax
        P = Diagonal(SVector(p1, p2))
        G = inv(I + P * V) * P
        proba = weight_dist(p1, p2)

        m̂ += (α * proba) * (G * SVector(1, 1))
        Q̂ += (α * proba) * (G * ((vstar + σ²) .+ B * Q * B') * G')
        V̂ += (α * proba) * G
    end

    return HatOverlaps(α * m̂, α * Q̂, α * V̂)
end

## State evolution

function state_evolution(
    ::Ridge;
    weight_dist::Function,
    α::Real,
    λ::Real,
    σ²::Real,
    ρ::Real=1.0,
    pmax=8,
    relative_tolerance=1e-4,
    max_iteration=100,
)
    overlaps, hatoverlaps = Overlaps(), HatOverlaps()
    
    for _ in 0:max_iteration
        new_hatoverlaps = update_hatoverlaps(overlaps; weight_dist, α, σ², ρ, pmax)
        new_overlaps = update_overlaps(new_hatoverlaps; λ)

        if relative_difference(new_overlaps, overlaps) < relative_tolerance
            return new_overlaps, new_hatoverlaps
        else
            overlaps, hatoverlaps = new_overlaps, new_hatoverlaps
        end
    end

    @warn "State evolution did not converge after $max_iteration iterations"
    return overlaps, hatoverlaps
end
