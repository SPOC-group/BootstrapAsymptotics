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

$(TYPEDFIELDS)
"""
@kwdef struct RidgeOverparametrized <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
    "correlation between sutdent and teacher features"
    κ1::Float64 = 1.0
    "white noise std. due to random features"
    κstar::Float64 = 0.0
    "Student of teacher dimension"
    student_over_teacher_dim::Float64 = 1.0
end

"""
$(TYPEDEF)

Bayes optimal setting for overparametrized random featruzres : λ is not used in this case. 
$(TYPEDFIELDS)
"""
@kwdef struct BayesOptimalRidgeOverparametrized <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    # "regularization strength"
    # λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
    "correlation between sutdent and teacher features"
    κ1::Float64 = 1.0
    "white noise std. due to random features"
    κstar::Float64 = 0.0
    "Student of teacher dimension"
    student_over_teacher_dim::Float64 = 1.0
end

function Base.show(io::IO, problem::Ridge)
    (; α, Δ, λ, ρ) = problem
    return print(io, "Ridge(α=$(round(α, sigdigits=3)), λ=$λ, ρ=$ρ, Δ=$Δ)")
end
