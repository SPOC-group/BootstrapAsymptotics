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

function sample_data(rng::AbstractRNG, ::Problem, n::Integer, d::Integer)
    X = randn(rng, n, d) ./ sqrt(d)
    w_star = randn(rng, d) ./ sqrt(d)
    return (; X, w_star)
end

function sample(rng::AbstractRNG, problem::Problem, n::Integer, d::Integer)
    (; X, w_star) = sample_data(rng, problem, n, d)
    y = sample_labels(rng, problem, X, w_star)
    return (; X, w_star, y)
end
