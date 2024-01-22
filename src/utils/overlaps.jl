@kwdef struct Overlaps{Vec1<:AbstractVector,Mat1<:AbstractMatrix,Mat2<:AbstractMatrix}
    m::Vec1
    Q::Mat1
    # TODO : V can be a vector (maybe it simplifies things ?)
    V::Mat2
end

@kwdef struct HatOverlaps{Vec1<:AbstractVector,Mat1<:AbstractMatrix,Mat2<:AbstractMatrix}
    m_hat::Vec1
    Q_hat::Mat1
    V_hat::Mat2
end

function init_overlaps()
    m = SVector(0.0, 0.0)
    Q = Symmetric(SMatrix{2,2}(1.0, 0.01, 0.01, 1.0))
    V = Diagonal(SVector(1.0, 1.0))
    return Overlaps(m, Q, V), HatOverlaps(copy(m), copy(Q), copy(V))
end

function relative_difference(overlaps::Overlaps, overlaps_ref::Overlaps)
    return norm(overlaps.Q - overlaps_ref.Q) / norm(overlaps.Q)
end
