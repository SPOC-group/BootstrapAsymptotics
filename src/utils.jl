bernpdf(r::Real, p::Integer) = isone(p) ? r : one(r) - r

function logistic_der(x::Real)
    s = logistic(x)
    return s * (1 - s)
end

logistic_loss(y::Real, z::Real) = log1pexp(-y * z)
logistic_loss_der(y::Real, z::Real) = -y * logistic(-y * z)
logistic_loss_der2(y::Real, z::Real) = y^2 * logistic_der(-y * z)

function get_additional_noise_from_kappas(kappa1, kappastar, student_dim_over_teacher_dim)
    kk1 = kappa1^2
    kkstar = kappastar^2
    lambda_minus = (1.0 - sqrt(student_dim_over_teacher_dim))^2
    lambda_plus = (1.0 + sqrt(student_dim_over_teacher_dim))^2
    
    # Function to integrate
    to_integrate(lambda_, kk1, kkstar, lambda_minus, lambda_plus) = 
        sqrt((lambda_plus - lambda_) * (lambda_ - lambda_minus)) / (kkstar + kk1 * lambda_)
    
    # Perform numerical integration
    integral, _ = quadgk(
        λ -> to_integrate(λ, kk1, kkstar, lambda_minus, lambda_plus), 
        lambda_minus, 
        lambda_plus
    )
    
    return 1.0 - kk1 * integral / (2 * π)
end