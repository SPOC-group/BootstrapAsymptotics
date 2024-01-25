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
