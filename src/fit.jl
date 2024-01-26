function fit(::Logistic, ::ERM, X, y)
    return w
end

function fit(::Ridge, ::ERM, X, y)
    w = (X' * X + λ * I) \ (X' * y)
    return w
end

function fit(problem::Problem, algo::ERM, X, y, w_star)
    return fit(problem, algo, X, y)
end

function fit(rng::AbstractRNG, problem::Problem, ::PairBootstrap, X, y, w_star)
    ind = sample(rng, length(y), length(y); replace=true)
    return fit(problem, ERM(), X[ind, :], y[ind])
end

function fit(rng::AbstractRNG, problem::Problem, algo::SubsamplingBootstrap, X, y, w_star)
    ind = rand(rng, length(y)) .< algo.r
    return fit(problem, ERM(), X[ind, :], y[ind])
end

function fit(rng::AbstractRNG, problem::Ridge, ::ResidualBootstrap, X, y, w_star)
    w_est = fit(problem, ERM(), X, y)
    y_pred = X * w_est
    Δ_est = mean(abs2, y - y_pred)
    new_y = sample_labels(rng, problem, X, w_est; Δ=Δ_est)
    return fit(problem, ERM(), X, new_y)
end

function fit(rng::AbstractRNG, problem::Logistic, ::ResidualBootstrap, X, y, w_star)
    w_est = fit(ERM(), X, y)
    new_y = sample_labels(rng, problem, X, w_est)
    return fit(problem, ERM(), X, new_y)
end

function fit(rng::AbstractRNG, problem::Problem, ::FullResampling, X, y, w_star)
    n, d = size(X)
    new_X = sample_data(rng, problem, n, d)
    new_y = sample_labels(rng, problem, X, w_star)
    return fit(problem, ERM(), new_X, new_y)
end

function fit(rng::AbstractRNG, problem::Problem, ::LabelResampling, X, y, w_star)
    new_y = sample_labels(rng, problem, X, w_star)
    return fit(problem, ERM(), X, new_y)
end
