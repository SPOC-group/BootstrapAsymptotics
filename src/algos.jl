abstract type Algorithm end

abstract type WeightBased <: Algorithm end
abstract type LabelBased <: Algorithm end

struct ERM <: Algorithm end

struct FullResampling <: WeightBased end

@kwdef struct PairBootstrap <: WeightBased
    p_max::Int = 8
end

@kwdef struct SubsamplingBootstrap <: WeightBased
    r::Float64 = 1.0
end

struct LabelResampling <: LabelBased end
struct ResidualBootstrap <: LabelBased end

## Labels

discrete_labels(::WeightBased, ::WeightBased) = (-1, +1)

function discrete_labels(::LabelBased, ::LabelBased)
    return (SVector(-1, -1), SVector(+1, -1), SVector(-1, +1), SVector(+1, +1))
end

## Dataset size

size_mul(::WeightBased, ::WeightBased) = 1
size_mul(::FullResampling, ::FullResampling) = 2

## Weight ranges

weight_range(::LabelBased) = 1:1
weight_range(::WeightBased) = 0:1
weight_range(algo::PairBootstrap) = 0:(algo.p_max)

function weight_range(algo1::WeightBased, algo2::WeightBased)
    return product(weight_range(algo1), weight_range(algo2))
end

## Weight distributions

# Single

bernpdf(r::Real, p::Integer) = isone(p) ? r : one(r) - r

weight_dist(::LabelBased, p::Integer) = isone(p)
weight_dist(::PairBootstrap, p::Integer) = poispdf(1, p)
weight_dist(algo::SubsamplingBootstrap, p::Integer) = bernpdf(algo.r, p)

function weight_dist(algo1::Algorithm, algo2::Algorithm, p1::Integer, p2::Integer)
    return weight_dist(algo1, p1) * weight_dist(algo2, p2)
end

function weight_dist(::FullResampling, ::FullResampling, p1::Integer, p2::Integer)
    return ((isone(p1) && iszero(p2)) + (iszero(p1) && isone(p2))) / 2
end
