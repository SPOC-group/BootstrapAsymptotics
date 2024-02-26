abstract type Problem end

"""
$(TYPEDEF)

Logistic regression problem with ridge penalty.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Logistic <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
end

function Base.show(io::IO, problem::Logistic)
    (; α, λ, ρ) = problem
    return print(io, "Logistic(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ)")
end

"""
$(TYPEDEF)

Least squares regression problem with ridge penalty.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Ridge <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
end

function Base.show(io::IO, problem::Ridge)
    (; α, Δ, λ, ρ) = problem
    return print(io, "Ridge(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ, Δ=$Δ)")
end
