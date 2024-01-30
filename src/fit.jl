function StatsAPI.fit(::Logistic, ::ERM, X::AbstractMatrix, y::AbstractVector)
    return error("not implemented")
end

function StatsAPI.fit(problem::Ridge, ::ERM, X::AbstractMatrix, y::AbstractVector)
    w = (X' * X + problem.λ * I) \ (X' * y)
    return w
end

function StatsAPI.fit(
    ::AbstractRNG, problem::Problem, algo::Algorithm, X::AbstractMatrix, y::AbstractVector
)
    return fit(problem, algo, X, y)
end

function StatsAPI.fit(
    rng,
    ::AbstractRNG,
    problem::Problem,
    algo::ERM,
    X::AbstractMatrix,
    y::AbstractVector,
    w_star::AbstractVector,
)
    return fit(rng, problem, algo, X, y)
end

function StatsAPI.fit(
    rng::AbstractRNG,
    problem::Problem,
    ::PairBootstrap,
    X::AbstractMatrix,
    y::AbstractVector,
)
    n = length(y)
    ind = sample(rng, 1:n, n; replace=true)
    return @views fit(problem, ERM(), X[ind, :], y[ind])
end

function StatsAPI.fit(
    rng::AbstractRNG,
    problem::Problem,
    algo::SubsamplingBootstrap,
    X::AbstractMatrix,
    y::AbstractVector,
)
    n = length(y)
    ind = rand(rng, n) .< algo.r
    return @views fit(problem, ERM(), X[ind, :], y[ind])
end

function StatsAPI.fit(
    rng::AbstractRNG,
    problem::Problem,
    ::FullResampling,
    X::AbstractMatrix,
    y::AbstractVector,
    w_star::AbstractVector,
)
    n = length(y)
    new_X = sample_data(rng, problem, n)
    new_y = sample_labels(rng, problem, X, w_star)
    return fit(problem, ERM(), new_X, new_y)
end

function StatsAPI.fit(
    rng::AbstractRNG,
    problem::Problem,
    ::LabelResampling,
    X::AbstractMatrix,
    y::AbstractVector,
    w_star::AbstractVector,
)
    new_y = sample_labels(rng, problem, X, w_star)
    return fit(problem, ERM(), X, new_y)
end
