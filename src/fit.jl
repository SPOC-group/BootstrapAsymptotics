## ERM

function StatsAPI.fit(problem::Logistic, ::ERM, X::AbstractMatrix, y::AbstractVector)
    (; λ) = problem
    model = MLJLinearModels.LogisticRegression(
        λ; fit_intercept=false, scale_penalty_with_samples=false
    )
    w = MLJLinearModels.fit(model, X, y)
    return w
end

function StatsAPI.fit(problem::Ridge, ::ERM, X::AbstractMatrix, y::AbstractVector)
    (; λ) = problem
    model = MLJLinearModels.RidgeRegression(
        λ; fit_intercept=false, scale_penalty_with_samples=false
    )
    w = MLJLinearModels.fit(model, X, y)
    return w
end

## Fallbacks

function StatsAPI.fit(
    ::AbstractRNG, problem::Problem, algo::Algorithm, X::AbstractMatrix, y::AbstractVector
)
    return fit(problem, algo, X, y)
end

function StatsAPI.fit(
    rng::AbstractRNG,
    problem::Problem,
    algo::Algorithm,
    X::AbstractMatrix,
    y::AbstractVector,
    w_star::AbstractVector,
)
    return fit(rng, problem, algo, X, y)
end

## Resampling

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
    new_y = sample_labels(rng, problem, new_X, w_star)
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

## Bayes optimal

function StatsAPI.fit(
    rng::AbstractRNG, ::Ridge, algo::BayesOpt, X::AbstractMatrix, y::AbstractVector;
)
    c = Turing.sample(
        rng,
        model_ridge(X, y),
        algo.sampler,
        algo.nb_samples;
        drop_warmup=true,
        verbose=false,
        progress=false,
    )
    w_samples = group(c, :w).value
    w_mean = dropdims(mean(w_samples; dims=(1, 3)); dims=(1, 3))
    return Vector(w_mean)
end

function StatsAPI.fit(
    rng::AbstractRNG, ::Logistic, algo::BayesOpt, X::AbstractMatrix, y::AbstractVector;
)
    y_bin = Bool.((y .+ 1) .÷ 2)
    c = Turing.sample(
        rng,
        model_logistic(X, y_bin),
        algo.sampler,
        algo.nb_samples;
        drop_warmup=true,
        verbose=false,
        progress=false,
    )
    w_samples = group(c, :w).value
    w_mean = dropdims(mean(w_samples; dims=(1, 3)); dims=(1, 3))
    return Vector(w_mean)
end
