function sample_labels(rng::AbstractRNG, ::Logistic, X::AbstractMatrix, w::AbstractVector)
    y = sign.(X * w)
    return y
end
