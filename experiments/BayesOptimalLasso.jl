using QuadGK
using HCubature
using SpecialFunctions: erfc, erf

function Iplus(b::Real, A::Real)::Real
    function integrand(x::Real)
        return exp(-abs(x) + b * x - (A * x^2 / 2))
    end

    return quadgk(integrand, 0, Inf)[1]

end

function Iplus_closed_form(b::Real, A::Real)::Real
    b = b - 1.0 # change of var. : -1.0 because we integrage on negative x 
    return exp(b^2 / (2 * A)) * sqrt(π / (2 * A)) * erfc(- b / sqrt(2 * A))
end

### 

function Iminus(b::Real, A::Real)::Real
    function integrand(x::Real)
        return exp(-abs(x) + b * x - (A * x^2 / 2))
    end

    return quadgk(integrand, -Inf, 0)[1]
end

function Iminus_closed_form(b::Real, A::Real)::Real
    b = b + 1.0 # we integrate on positive x
    return exp(b^2 / (2 * A)) * sqrt(π /  (2 * A)) * erfc( b / sqrt(2 * A))
end

b, A = 10.0, 5.0
println(Iplus(b, A), " ", Iplus_closed_form(b, A) )
println(Iminus(b, A), " ", Iminus_closed_form(b, A))