using LinearAlgebra

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
    w_est          = fit(problem, ERM(), X, y)

    w_samples      = [ fit(rng, problem, algo, X, y, w) for k in 1:K ]
    w_samples_mean = mean(w_samples)

    bias2          = norm(w_samples_mean - w_est)^2 / d
    variance       = mean(norm(w_samples[k] - w_samples_mean)^2 for k in 1:K) / d
    return bias2, variance
end

function bias_variance_empirical(
    rng::AbstractRNG, problem::Problem, algo::ResidualBootstrap; n::Integer, K::Integer
)
    (; X, y, w) = sample_all(rng, problem, n)
    d = size(X, 2)
    w_est = fit(problem, ERM(), X, y)

    if problem isa Ridge
        problem_residual = Ridge(;
            ρ=norm(w_est)^2 / d, α=problem.α, λ=problem.λ, Δ=norm(y - X * w_est)^2 / (n - 1)
        )
    else
        problem_residual = Logistic(; ρ=norm(w_est)^2 / d, α=problem.α, λ=problem.λ)
    end

    w_samples = [ fit(rng, problem_residual, LabelResampling(), X, y, w_est) for k in 1:K]
    w_samples_mean = mean(w_samples)
    
    bias2    = norm(w_samples_mean - w_est)^2 / d
    variance = mean(norm(w_samples[k] - w_samples_mean)^2 for k in 1:K) / d
    return bias2, variance
end

function bias_variance_empirical(
    rng::AbstractRNG, problem::Ridge, algo::BayesOpt; n::Integer, K::Integer
)
    # LC : Stupid question but is the Bayes-optimal really unbiased ?
    # TODO : So far we don't return anything for the bias but we should implement it 
    @assert problem.ρ == 1.0 && problem.Δ == 1.0 && problem.λ == 1.0

    (; X, y, w) = sample_all(rng, problem, n)
    d = floor(n / problem.α)
    bias     = Inf
    # here this is the variance w.r.t the Posterior distribution (and not w.r.t the resampling of D)
    variance = LinearAlgebra.tr(inv(X'X + problem.λ * I)) / d
    return bias, variance
end

function bias_variance_empirical(
    rng::AbstractRNG, problem::Logistic, algo::BayesOpt; n::Integer, K::Integer
)
    @assert problem.ρ == 1.0 && problem.λ == 1.0
    (; X, y, w) = sample_all(rng, problem, n)
    d = floor(n / problem.α)

    xhat, vhat = gamp(problem, X, y)
    
    bias     = Inf
    # here this is the variance w.r.t the Posterior distribution (and not w.r.t the resampling of D)
    variance = mean(vhat)

    return bias, variance
end

## The functions below are not used apart from testing the codebase

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

function variance_state_evolution(
    problem::Problem, algo::ResidualBootstrap; check_convergence::Bool=true, kwargs...
)
    result_erm = state_evolution(problem, FullResampling(), FullResampling(); kwargs...)
    if check_convergence
        @assert result.stats.converged
    end

    # check if problem is instance of Ridge
    if problem isa Ridge
        problem_residual = Ridge(;
            ρ=result_erm.overlaps.Q[1, 1],
            α=problem.α,
            λ=problem.λ,
            Δ=(
                problem.ρ + result_erm.overlaps.Q[1, 1] - 2 * result_erm.overlaps.m[1] +
                problem.Δ
            ) / (1.0 + result_erm.overlaps.V[1])^2.0,
        )
    else
        problem_residual = Logistic(;
            ρ=result_erm.overlaps.Q[1, 1], α=problem.α, λ=problem.λ
        )
    end

    result = state_evolution(
        problem_residual, LabelResampling(), LabelResampling(); kwargs...
    )

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
