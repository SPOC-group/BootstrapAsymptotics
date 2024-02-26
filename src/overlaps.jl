"""
$(TYPEDEF)

Overlap or hat overlap storage for state evolution between two algorithms.

# Fields

$(TYPEDFIELDS)
"""
struct Overlaps{hat,T1<:AbstractVector,T2<:AbstractMatrix,T3<:AbstractMatrix}
    m::T1
    Q::T2
    V::T3

    function Overlaps{hat}(m::T1, Q::T2, V::T3) where {hat,T1,T2,T3}
        @assert length(m) == 2
        @assert size(Q, 1) == size(Q, 2) == 2
        @assert size(V, 1) == size(V, 2) == 2
        return new{hat,T1,T2,T3}(m, Q, V)
    end
end

function Base.show(io::IO, overlaps::Overlaps{hat}) where {hat}
    (; m, Q, V) = overlaps
    if hat
        return print(io, "HatOverlaps($m, $Q, $V)")
    else
        return print(io, "Overlaps($m, $Q, $V)")
    end
end

function Overlaps{hat}() where {hat}
    m = SVector(0.0, 0.0)
    Q = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0)
    V = Diagonal(SVector(1.0, 1.0))
    return Overlaps{hat}(m, Q, V)
end

function close_enough(
    overlaps::Overlaps{hat}, overlaps_ref::Overlaps{hat}; rtol::Real
) where {hat}
    return (
        norm(overlaps.m - overlaps_ref.m, Inf) / norm(overlaps_ref.m, Inf) < rtol &&
        norm(overlaps.Q - overlaps_ref.Q, Inf) / norm(overlaps_ref.Q, Inf) < rtol &&
        norm(overlaps.V - overlaps_ref.V, Inf) / norm(overlaps_ref.V, Inf) < rtol
    )
end

function close_enough(overlap::Real, overlap_ref::Real; rtol::Real)
    return norm(overlap - overlap_ref, Inf) / norm(overlap_ref, Inf) < rtol
end
