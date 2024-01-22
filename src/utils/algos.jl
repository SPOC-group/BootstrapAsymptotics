abstract type Algorithm end
abstract type BootstrapAlgorithm <: Algorithm end
abstract type ResamplingAlgorithm <: Algorithm end

struct ERM <: Algorithm end

@kwdef struct PairBootstrap <: BootstrapAlgorithm
    p_max::Int
end

@kwdef struct ResidualBootstrap <: BootstrapAlgorithm
    p_max::Int
end

struct FullResampling <: ResamplingAlgorithm end
struct LabelResampling <: ResamplingAlgorithm end

## Weight ranges

# TODO: control error

function weight_range(algo::BootstrapAlgorithm)
    return 0:(algo.p_max)
end

## Weight distributions

function weight_dist(::PairBootstrap, p::Integer)
    return poispdf(1, p)
end

function weight_dist(::PairBootstrap, ::PairBootstrap, p1::Integer, p2::Integer)
    return poispdf(1, p1) * poispdf(1, p2)
end

function weight_dist(::PairBootstrap, ::FullResampling, p1::Integer, p2::Integer)
    return poispdf(1, p1) * isone(p2)
end

function weight_dist(::FullResampling, ::FullResampling, p1::Integer, p2::Integer)
    return ((isone(p1) && iszero(p2)) + (iszero(p1) && isone(p2))) / 2
end

function weight_dist(::LabelResampling, ::LabelResampling, p1::Integer, p2::Integer)
    return isone(p1) && isone(p2)
end
