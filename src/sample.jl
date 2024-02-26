"""
$(SIGNATURES)

Sample the data matrix `X` for a given `problem` with population size `n`.
"""
function sample_data(rng::AbstractRNG, problem::Problem, n::Integer)
    d = ceil(Int, n / problem.α)
    X = randn(rng, n, d) ./ sqrt(d)
    return X
end

"""
$(SIGNATURES)

Sample the weights vector `w` for a given `problem` with population size `n` (from which the dimension is deduced).
"""
function sample_weights(rng::AbstractRNG, problem::Problem, n::Integer)
    d = ceil(Int, n / problem.α)
    w = randn(rng, d)
    return w
end

"""
$(SIGNATURES)

Sample the labels vector `y` for a given `problem` from the features `X` and weights `w`.
"""
function sample_labels(rng::AbstractRNG, ::Logistic, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = 2 .* (rand(rng, n) .< logistic.(X * w)) .- 1
    return y
end

function sample_labels(
    rng::AbstractRNG, problem::Ridge, X::AbstractMatrix, w::AbstractVector;
)
    n = size(X, 1)
    y = X * w .+ sqrt.(problem.Δ) .* randn(rng, n)
    return y
end

"""
$(SIGNATURES)

Sample `X`, `w` and `y` all at once for a given `problem` with population size `n`.
"""
function sample_all(rng::AbstractRNG, problem::Problem, n::Integer)
    X = sample_data(rng, problem, n)
    w = sample_weights(rng, problem, n)
    y = sample_labels(rng, problem, X, w)
    return (; X, w, y)
end
