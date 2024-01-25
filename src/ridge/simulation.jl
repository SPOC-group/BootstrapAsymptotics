function sample_labels(rng::AbstractRNG, ::Ridge, X::AbstractMatrix, w::AbstractVector)
    y = X * w .+ randn(rng, n)
    return y
end

function fit(::Ridge, ::ERM, X::AbstractMatrix, y::AbstractVector)
    w = (X' * X + λ * I) \ (X' * y)
    return w
end
