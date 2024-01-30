bernpdf(r::Real, p::Integer) = isone(p) ? r : one(r) - r

function logistic_der(x::Real)
    s = logistic(x)
    return s * (1 - s)
end

logistic_loss(y::Real, z::Real) = log1pexp(-y * z)
logistic_loss_der(y::Real, z::Real) = -y * logistic(-y * z)
logistic_loss_der2(y::Real, z::Real) = y^2 * logistic_der(-y * z)
