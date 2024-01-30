abstract type Problem end

@kwdef struct Logistic <: Problem
    α::Float64 = 1.0
    λ::Float64 = 1.0
    ρ::Float64 = 1.0
end

function Base.show(io::IO, problem::Logistic)
    (; α, λ, ρ) = problem
    return print(io, "Logistic(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ)")
end

@kwdef struct Ridge <: Problem
    α::Float64 = 1.0
    Δ::Float64 = 1.0
    λ::Float64 = 1.0
    ρ::Float64 = 1.0
end

function Base.show(io::IO, problem::Ridge)
    (; α, Δ, λ, ρ) = problem
    return print(io, "Logistic(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ, Δ=$Δ)")
end
