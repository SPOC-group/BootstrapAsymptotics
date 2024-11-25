abstract type Problem end

"""
$(TYPEDEF)

Logistic regression problem with ridge penalty.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Logistic <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
end

function Base.show(io::IO, problem::Logistic)
    (; α, λ, ρ) = problem
    return print(io, "Logistic(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ)")
end

"""
$(TYPEDEF)

Least squares regression problem with ridge penalty.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Ridge <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
end

"""
$(TYPEDEF)

Ridge regression problem with a random feature model : additional fields are κ1 and κ* to model the kernel.
ρ is still the teacher norm and Δ the initial teacher noise variance. (ρ + Δ is left unchanged in the channel)

NOTE : In the code, we assume that the two learners have different random features from the same distribution

TODO : Allow to use the same random feature and combine with resampling (will be useful for variance analysis)

$(TYPEDFIELDS)
"""
@kwdef struct RidgeOverparametrized <: Problem
    "ratio of population over student dimension `n/p`"
    α::Float64
    "Gaussian noise variance"
    Δ::Float64
    "regularization strength"
    λ::Float64
    "teacher weight"
    ρ::Float64
    "correlation between sutdent and teacher features"
    κ1::Float64
    "white noise std. due to random features"
    κstar::Float64
    "Student of teacher dimension"
    student_over_teacher_dim::Float64 
end

# function to take into account the additional noise coming from the random features
function build_ridge_overparametrized(;
    α::Float64,
    true_Δ::Float64,
    λ::Float64,
    true_ρ::Float64,
    κ1::Float64,
    κstar::Float64,
    student_over_teacher_dim::Float64 
)::RidgeOverparametrized
    Δ_add = true_ρ * get_additional_noise_from_kappas(κ1, κstar, student_over_teacher_dim)
    return RidgeOverparametrized(
        α     = α,
        Δ     = true_Δ + Δ_add,
        κ1    = κ1,
        κstar = κstar,
        ρ = true_ρ - Δ_add,
        λ = λ,
        student_over_teacher_dim = student_over_teacher_dim
    )
end

"""
$(TYPEDEF)

Bayes optimal setting for overparametrized random featruzres : λ is not used in this case. 
$(TYPEDFIELDS)
"""
@kwdef struct BayesOptimalRidgeOverparametrized <: Problem
    "ratio of population over student dimension `n/p`"
    α::Float64
    "correlation between sutdent and teacher features"
    κ1::Float64 
    "white noise std. due to random features"
    κstar::Float64
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    # "regularization strength"
    # λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
    "Student of teacher dimension"
    student_over_teacher_dim::Float64 = 1.0
end

function Base.show(io::IO, problem::Ridge)
    (; α, Δ, λ, ρ) = problem
    return print(io, "Ridge(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ, Δ=$Δ)")
end


@kwdef struct KernelRidgeOverparametrized <: Problem
    "correlation between sutdent and teacher features"
    κ1::Float64
    "white noise std. due to random features"
    κstar::Float64 
    "ratio of population over teacher dimension `n/d`"
    δ::Float64
    "reg."
    λ::Float64
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    # "regularization strength"
    # λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
end