abstract type Algorithm end

struct ERM <: Algorithm end
struct FullResampling <: Algorithm end
struct LabelResampling <: Algorithm end

@kwdef struct PairBootstrap <: Algorithm
    p_max::Int = 8
end

@kwdef struct Subsampling <: Algorithm
    r::Float64 = 1.0
end

@kwdef struct ResidualBootstrap <: Algorithm end

@kwdef struct BayesOpt{S} <: Algorithm
    sampler::S = NUTS(; adtype=AutoReverseDiff())
    nb_samples::Int = 100
end

## Labels

same_labels(::Algorithm, ::Algorithm) = true
same_labels(::LabelResampling, ::LabelResampling) = false

## Weight ranges

weight_range(::Algorithm) = 0:1
weight_range(algo::PairBootstrap) = 0:(algo.p_max)

## Weight distributions

weight_dist(::BayesOpt, p::Integer) = error("No state evolution for BayesOpt")
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
