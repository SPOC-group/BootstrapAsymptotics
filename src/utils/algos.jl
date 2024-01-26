abstract type Algorithm end
abstract type BootstrapAlgorithm <: Algorithm end

struct ERM <: Algorithm end
struct FullResampling <: Algorithm end
struct LabelResampling <: Algorithm end

@kwdef struct PairBootstrap <: Algorithm
    p_max::Int = 8
end

@kwdef struct ResidualBootstrap <: Algorithm
    p_max::Int = 8
end

@kwdef struct SubsamplingBootstrap <: Algorithm
    r::Float64 = 1.0
end

## Weight ranges

# TODO: control error

weight_range(::Algorithm) = 0:1
weight_range(algo::PairBootstrap) = 0:(algo.p_max)
weight_range(algo::ResidualBootstrap) = 0:(algo.p_max)

## Weight distributions

# Single

weight_dist(::Algorithm, p::Integer) = isone(p)
weight_dist(::PairBootstrap, p::Integer) = poispdf(1, p)

function weight_dist(algo::SubsamplingBootstrap, p::Integer)
    return isone(p) ? algo.r : one(algo.r) - algo.r
end

# Double

function weight_dist(algo1::Algorithm, algo2::Algorithm, p1::Integer, p2::Integer)
    return weight_dist(algo1, p1) * weight_dist(algo2, p2)
end

function weight_dist(::FullResampling, ::FullResampling, p1::Integer, p2::Integer)
    return ((isone(p1) && iszero(p2)) + (iszero(p1) && isone(p2)))  # sums to 2
end
