@model function model_ridge(X::AbstractMatrix{T1}, y::AbstractVector{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    n, d = size(X)
    w ~ filldist(Normal(zero(T), one(T)), d)
    X ~ filldist(Normal(zero(T), one(T) / sqrt(d)), n, d)
    y ~ MvNormal(X * w, I)
    return w
end

@model function model_logistic(
    X::AbstractMatrix{T1}, y_bin::AbstractVector{T2}
) where {T1,T2}
    T = promote_type(T1, T2)
    n, d = size(X)
    w ~ filldist(Normal(zero(T), one(T)), d)
    X ~ filldist(Normal(zero(T), one(T) / sqrt(d)), n, d)
    y_bin ~ arraydist(Bernoulli.(logistic.(X * w)))
    return w
end
