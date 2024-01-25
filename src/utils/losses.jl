function logistic_der(x::Real)
    s = logistic(x)
    return s * (1 - s)
end

logistic_loss(y, z) = log1p(exp(-y * z))
logistic_loss_der(y, z) = -y * logistic(-y * z)
logistic_loss_der2(y, z) = y^2 * logistic_der(-y * z)
