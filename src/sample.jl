function sample_data(rng::AbstractRNG, ::Problem, n::Integer, d::Integer)
    X = randn(rng, n, d) ./ sqrt(d)
    return X
end

function sample_weights(rng::AbstractRNG, ::Problem, d::Integer)
    w = randn(rng, d) ./ sqrt(d)
    return w
end

function sample_labels(rng::AbstractRNG, ::Logistic, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = 2 .* (rand(rng, n) .< logistic.(X * w)) .- 1
    return y
end

function sample_labels(
    rng::AbstractRNG, ::Ridge, X::AbstractMatrix, w::AbstractVector; Δ::Real=1
)
    n = size(X, 1)
    y = X * w .+ Δ .* randn(rng, n)
    return y
end

function fit(::Ridge, ::ERM, X::AbstractMatrix, y::AbstractVector)
    w = (X' * X + λ * I) \ (X' * y)
    return w
end
