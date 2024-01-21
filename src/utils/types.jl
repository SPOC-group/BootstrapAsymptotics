abstract type Problem end

struct Logistic <: Problem end
struct Ridge <: Problem end

abstract type Algorithm end

struct Bootstrap <: Algorithm end
struct FullResampling <: Algorithm end
struct TargetResampling <: Algorithm end

struct Overlaps{Vec1<:AbstractVector,Mat1<:AbstractMatrix,Mat2<:AbstractMatrix}
    m::Vec1
    Q::Mat1
    V::Mat2
end

struct HatOverlaps{Vec1<:AbstractVector,Mat1<:AbstractMatrix,Mat2<:AbstractMatrix}
    m_hat::Vec1
    Q_hat::Mat1
    V_hat::Mat2
end

function Overlaps()
    m = SVector(0.0, 0.0)
    Q = SMatrix{2,2}(1.0, 0.01, 0.01, 1.0)
    V = Diagonal(SVector(1.0, 1.0))
    return Overlaps(m, Q, V)
end

function HatOverlaps()
    m_hat = SVector(0.0, 0.0)
    Q_hat = SMatrix{2,2}(1.0, 0.01, 0.01, 1.0)
    V_hat = Diagonal(SVector(1.0, 1.0))
    return HatOverlaps(m_hat, Q_hat, V_hat)
end

function relative_difference(overlaps::Overlaps, overlaps_ref::Overlaps)
    return norm(overlaps.Q - overlaps_ref.Q) / norm(overlaps.Q)
end
