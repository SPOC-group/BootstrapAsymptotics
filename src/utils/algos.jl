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

@kwdef struct Subsampling <: BootstrapAlgorithm
    proba::Real
end

struct FullResampling <: ResamplingAlgorithm end
struct LabelResampling <: ResamplingAlgorithm end

## Weight ranges

# TODO: control error

function weight_range(algo::PairBootstrap)
    return 0:(algo.p_max)
end

function weight_range(algo::Subsampling)
    return 0:1
end

function weight_range(algo::FullResampling)
    return 0:1
end

function weight_range(algo::LabelResampling)
    return 1:1
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

function weight_dist(algo1::Subsampling, algo2::Subsampling, p1::Integer, p2::Integer)
    return (algo1.proba * isone(p1) + (1.0 - algo1.proba) * iszero(p1)) * (algo2.proba * isone(p2) + (1.0 - algo2.proba) * iszero(p2))
end

function weight_dist(algo1::Subsampling, ::FullResampling, p1::Integer, p2::Integer)
    return (algo1.proba * isone(p1) + (1.0 - algo1.proba) * iszero(p1)) * isone(p2)
end
