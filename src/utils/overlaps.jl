struct Overlaps{N,T1,T2,T3}
    m::T1
    Q::T2
    V::T3

    function Overlaps{1,T1,T2,T3}(m::Real, Q::Real, V::Real) where {T1,T2,T3}
        return new{1,T1,T2,T3}(m, Q, V)
    end

    function Overlaps{N,T1,T2,T3}(
        m::AbstractVector, Q::AbstractMatrix, V::AbstractMatrix
    ) where {N,T1,T2,T3}
        @assert N == length(m)
        @assert N == size(Q, 1) == size(Q, 2)
        @assert N == size(V, 1) == size(V, 2)
        return new{N,T1,T2,T3}(m, Q, V)
    end
end

function Overlaps(m::T1, Q::T2, V::T3) where {T1<:Real,T2<:Real,T3<:Real}
    return Overlaps{1,T1,T2,T3}(m, Q, V)
end

function Overlaps(
    m::T1, Q::T2, V::T3
) where {T1<:AbstractVector,T2<:AbstractMatrix,T3<:AbstractMatrix}
    N = length(m)
    return Overlaps{N,T1,T2,T3}(m, Q, V)
end

function init_overlaps(::Val{1})
    m = 0.0
    Q = 1.0
    V = 1.0
    return Overlaps(m, Q, V)
end

function init_overlaps(::Val{2})
    m = SVector(0.0, 0.0)
    Q = SMatrix{2,2}(1.0, 0.01, 0.01, 1.0)
    V = Diagonal(SVector(1.0, 1.0))
    return Overlaps(m, Q, V)
end

function close_enough(overlaps::Overlaps, overlaps_ref::Overlaps; rtol::Real)
    return norm(overlaps.Q - overlaps_ref.Q) / norm(overlaps_ref.Q) < rtol
end
