function check_get_n_from_teacher_dim(problem::Problem; n::Union{Nothing, Integer} = nothing, teacher_dim::Union{Nothing, Integer} = nothing)::Integer
    if n === nothing
        if teacher_dim === nothing
            error("Must set n or teacher_dim")
        else 
            if problem isa RidgeOverparametrized
                return ceil(Int, teacher_dim * problem.student_over_teacher_dim * problem.α)
            else
                return ceil(Int, teacher_dim * problem.α)
            end
        end
    else
        return n
    end
end

"""
$(SIGNATURES)

Sample the data matrix `X` for a given `problem` with population size `n`.
"""
function sample_data(rng::AbstractRNG, problem::Problem; teacher_dim::Integer, n::Union{Nothing, Integer} = nothing)
    n = check_get_n_from_teacher_dim(problem; n = n, teacher_dim = teacher_dim)
    
    X = randn(rng, n, teacher_dim) ./ sqrt(teacher_dim)
    return X
end

"""
$(SIGNATURES)

Sample the weights vector `w` for a given `problem` with population size `n` (from which the dimension is deduced).
"""
function sample_weights(rng::AbstractRNG, problem::Problem, n::Union{Nothing, Integer} = nothing; d::Union{Nothing, Integer} = nothing)
    if d === nothing
        if n === nothing
            error("Must set d or n")
        else
            d = ceil(Int, n / problem.α)
        end
    end

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
    rng::AbstractRNG, problem::Union{Ridge, RidgeOverparametrized}, X::AbstractMatrix, w::AbstractVector;
)
    n = size(X, 1)
    y = X * w .+ sqrt.(problem.Δ) .* randn(rng, n)
    return y
end

"""
$(SIGNATURES)

Sample `X`, `w` and `y` all at once for a given `problem` with population size `n`.
"""
function sample_all(rng::AbstractRNG, problem::Problem; n::Union{Nothing, Integer} = nothing, teacher_dim::Integer)
    n = check_get_n_from_teacher_dim(problem; n=n, teacher_dim=teacher_dim)

    X = sample_data(rng, problem; n = n, teacher_dim = teacher_dim)
    w = sample_weights(rng, problem; d = teacher_dim)
    y = sample_labels(rng, problem, X, w)
    return (; X, w, y)
end

# for overparametrized random features 

function sample_data_fixed_teacher_dim(rng::AbstractRNG, problem::Problem, teacher_dim::Integer)
    student_dim = ceil(Int, teacher_dim * problem.student_over_teacher_dim)
    n = ceil(Int, problem.α * student_dim)

    X = randn(rng, n, teacher_dim) ./ sqrt(teacher_dim)
    return X
end

function sample_random_projection(rng::AbstractRNG, problem::RidgeOverparametrized, teacher_dim::Integer)
    student_dim = ceil(Int, teacher_dim * problem.student_over_teacher_dim)
    F = randn(rng, student_dim, teacher_dim)
    return F
end