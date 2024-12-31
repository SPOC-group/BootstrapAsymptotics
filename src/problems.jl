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
    "stzudent noise variance in the ridge loss"
    Δ̂::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight"
    ρ::Float64 = 1.0
end

@kwdef struct Lasso <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    "stzudent noise variance in the ridge loss"
    Δ̂::Float64 = 1.0
    "regularization strength"
    λ::Float64 = 1.0
    "teacher weight is 2 for lasso because of the laplace distribution"
    ρ::Float64 = 2.0
end

@kwdef struct LassoWithLassoTeacher <: Problem
    "ratio of population over dimension `n/d`"
    α::Float64 = 1.0
    "Gaussian noise variance"
    Δ::Float64 = 1.0
    "stzudent noise variance in the ridge loss"
    Δ̂::Float64 = 1.0
    "regularization strength is 1 in the BO case because of the Laplace distribution"
    λ::Float64 = 1.0
    "teacher weight is 2 for lasso because of the laplace distribution"
    ρ::Float64 = 2.0
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
    "stzudent noise variance in the ridge loss"
    Δ̂::Float64 = 1.0
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
    Δ::Float64
    Δ̂::Float64
    "teacher weight"
    ρ::Float64 # after the projection 
    true_ρ::Float64 # before the projection 
    "Student of teacher dimension"
    student_over_teacher_dim::Float64
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
    Δ::Float64
    "teacher weight"
    ρ::Float64
    "Gaussian noise variance"
    Δ̂::Float64 = 1.0
end

# 

@kwdef struct LogisticOverparametrized <: Problem
    "Additive noise coming from the overparametrization"
    Δ_add::Float64
    "ratio of population over student dimension `n/p`"
    α::Float64
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

# ==== 
function build_ridge_overparametrized(;
    α::Float64,
    true_Δ::Float64,
    λ::Union{Nothing, Float64}, # useless for Bayes optimal
    true_ρ::Float64,
    κ1::Float64,
    κstar::Float64,
    student_over_teacher_dim::Float64,
    Δ̂::Union{Nothing, Float64} = nothing,
    base_optimal::Bool = false
)::Union{RidgeOverparametrized, BayesOptimalRidgeOverparametrized}
    Δ_add = true_ρ * get_additional_noise_from_kappas(κ1, κstar, student_over_teacher_dim)
    
    if !base_optimal
        if λ === nothing || Δ̂ === nothing
            error("Must specify λ and Δ̂ for ERM estimator")
        end
        return RidgeOverparametrized(
            α     = α,
            Δ     = true_Δ + Δ_add,
            Δ̂     = Δ̂,
            κ1    = κ1,
            κstar = κstar,
            ρ     = true_ρ - Δ_add,
            λ     = λ,
            student_over_teacher_dim = student_over_teacher_dim
        )
    else
        return BayesOptimalRidgeOverparametrized(
            α     = α,
            Δ     = true_Δ + Δ_add,
            Δ̂     = true_Δ + Δ_add,
            κ1    = κ1,
            κstar = κstar,
            ρ = true_ρ - Δ_add,
            true_ρ = true_ρ,
            student_over_teacher_dim = student_over_teacher_dim
        )
    end
end

### 

function build_logistic_overparametrized(;
    α::Float64,
    true_Δ::Float64,
    λ::Float64, # useless for Bayes optimal
    true_ρ::Float64,
    κ1::Float64,
    κstar::Float64,
    student_over_teacher_dim::Float64,
)::LogisticOverparametrized
    Δ_add = true_ρ * get_additional_noise_from_kappas(κ1, κstar, student_over_teacher_dim)
    
    return LogisticOverparametrized(
        α     = α,
        Δ_add = Δ_add,
        κ1    = κ1,
        κstar = κstar,
        ρ     = true_ρ - Δ_add,
        λ     = λ,
        student_over_teacher_dim = student_over_teacher_dim
    )
end