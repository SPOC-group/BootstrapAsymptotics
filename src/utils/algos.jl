abstract type Algorithm end

@kwdef struct PairBootstrap <: Algorithm
    p_max::Int
end

@kwdef struct ResidualBootstrap <: Algorithm
    p_max::Int
end

struct FullResampling <: Algorithm end
struct LabelResampling <: Algorithm end

## Weight ranges

# TODO: control error

function weight_ranges(algo1::PairBootstrap, algo2::PairBootstrap)
    return 0:(algo1.p_max), 0:(algo2.p_max)
end

function weight_ranges(algo1::PairBootstrap, ::FullResampling)
    return 0:(algo1.p_max), 1:1
end

function weight_ranges(::FullResampling, ::FullResampling)
    return 0:1, 0:1
end

function weight_ranges(::LabelResampling, ::LabelResampling)
    return 1:1, 1:1
end

## Weight distributions

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
