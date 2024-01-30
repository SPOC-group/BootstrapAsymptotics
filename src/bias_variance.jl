function bias_variance_true(
    rng::AbstractRNG, problem::Problem; n::Integer, K::Integer, conditional=false
)
    (; X, y, w) = sample_all(rng, problem, n)
    d = size(X, 2)
    if conditional
        w_samples = [fit(rng, problem, LabelResampling(), X, y, w) for k in 1:K]
    else
        w_samples = [fit(rng, problem, FullResampling(), X, y, w) for k in 1:K]
    end
    w_samples_mean = mean(w_samples)
    bias2 = norm(w_samples_mean - w)^2 / d
    variance = mean(norm(w_samples[k] - w_samples_mean)^2 for k in 1:K) / d
    return bias2, variance
end

function bias_variance_empirical(
    rng::AbstractRNG, problem::Problem, algo::Algorithm; n::Integer, K::Integer
)
    (; X, y, w) = sample_all(rng, problem, n)
    d = size(X, 2)
    w_est = fit(problem, ERM(), X, y)
    w_samples = [fit(rng, problem, algo, X, y, w) for k in 1:K]
    w_samples_mean = mean(w_samples)
    bias2 = norm(w_samples_mean - w_est)^2 / d
    variance = mean(norm(w_samples[k] - w_samples_mean)^2 for k in 1:K) / d
    return bias2, variance
end

function variance_state_evolution(
    problem::Problem, algo::Algorithm; check_convergence::Bool=true, kwargs...
)
    result = state_evolution(problem, algo, algo; kwargs...)
    if check_convergence
        @assert result.stats.converged
    end
    variance = result.overlaps.Q[1, 1] - result.overlaps.Q[1, 2]
    return variance
end

function bias_state_evolution(
    problem::Problem, algo::Algorithm; check_convergence::Bool=true, kwargs...
)
    result_bootstrap = state_evolution(problem, algo, algo; kwargs...)
    result_mixed = state_evolution(problem, algo, FullResampling(); kwargs...)
    result_full = state_evolution(problem, FullResampling(), FullResampling(); kwargs...)
    if check_convergence
        @assert result_bootstrap.stats.converged
        @assert result_mixed.stats.converged
        @assert result_full.stats.converged
    end
    bias2 = (
        result_full.overlaps.Q[1, 1] + result_bootstrap.overlaps.Q[1, 2] -
        2 * result_mixed.overlaps.Q[1, 2]
    )
    return bias2
end
