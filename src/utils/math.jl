const LOGISTIC_PROBIT_FACTOR = 0.5875651988237005

indep_poisson(p1::Real, p2::Real) = poispdf(1, p1) * poispdf(1, p2)
