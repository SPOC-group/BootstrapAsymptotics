abstract type Problem end

@kwdef struct Logistic{R<:Real} <: Problem
    α::R = 1.0
    λ::R = 1.0
    ρ::R = 1.0
end

@kwdef struct Ridge{R<:Real} <: Problem
    α::R = 1.0
    Δ::R = 1.0
    λ::R = 1.0
    ρ::R = 1.0
end
