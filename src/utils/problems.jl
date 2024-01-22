abstract type Problem end

@kwdef struct Logistic{R<:Real} <: Problem
    α::R
    λ::R
    ρ::R
end

@kwdef struct Ridge{R<:Real} <: Problem
    α::R
    Δ::R
    λ::R
    ρ::R
end
