abstract type Algorithm end

"""
$(TYPEDEF)

Empirical Risk Minimization algorithm.
"""
struct ERM <: Algorithm end

"""
$(TYPEDEF)

Full resampling algorithm.
"""
struct FullResampling <: Algorithm end

"""
$(TYPEDEF)

Label resampling algorithm.
"""
struct LabelResampling <: Algorithm end

"""
$(TYPEDEF)

Standard (pair) bootstrap algorithm.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct PairBootstrap <: Algorithm
    "maximum weight for state evolution"
    p_max::Int = 8
end

"""
$(TYPEDEF)

Subsampling algorithm.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct Subsampling <: Algorithm
    "subsampling fraction"
    r::Float64 = 1.0
end

# Not useful in itself as it's LabelResampling using ERM but 
# keep it for consistency
"""
$(TYPEDEF)

Residual bootstrap algorithm, aka ERM + label resampling.
"""
@kwdef struct ResidualBootstrap <: Algorithm end

"""
$(TYPEDEF)

Bayes optimal estimation algorithm.
"""
@kwdef struct BayesOpt <: Algorithm end

## Labels

same_labels(::Algorithm, ::Algorithm) = true
same_labels(::LabelResampling, ::LabelResampling) = false

## Weight ranges

weight_range(::Algorithm) = 0:1
weight_range(algo::PairBootstrap) = 0:(algo.p_max)

## Weight distributions

weight_dist(::BayesOpt, p::Integer) = isone(p)
weight_dist(::Algorithm, p::Integer) = isone(p)
weight_dist(::PairBootstrap, p::Integer) = poispdf(1, p)
weight_dist(algo::Subsampling, p::Integer) = bernpdf(algo.r, p)

function weight_dist(algo1::Algorithm, algo2::Algorithm, p1::Integer, p2::Integer)
    return weight_dist(algo1, p1) * weight_dist(algo2, p2)
end

function weight_dist(::FullResampling, ::FullResampling, p1::Integer, p2::Integer)
    # warning: we include the factor 2 here
    return ((isone(p1) * iszero(p2)) + (iszero(p1) * isone(p2)))
end
