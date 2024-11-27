
### function for re-calibration of ensemble of models 

"""
compute the scaling to minimize the gaussian NLL

assume an infinite number of models on which we average, so that the variance is q_0 - q_1
and the rescaled variance is like (q₀ - q₁) / (scaling). So that the function to minimize is 
    
        NLL = 1/2 * (ρ - 2 × m + q + Δ) / ((q₀-q₁) / scaling) + 1 / 2 * log((q₀-q₁) / scaling)
    
    where q₀ and q₁ are the two diagonal elements of the Q matrix
"""
function get_scaling_gaussian_nll(overlaps::BootstrapAsymptotics.Overlaps{false}, problem::Problem)::Float64
    (; ρ, Δ) = problem
    
    # we have a closed form expression
    m = overlaps.m[1]
    q₁= overlaps.Q[1, 2]
    q₀ = overlaps.Q[1, 1]

    # when averaged, the MSE is ρ - 2 * m + q₁ and not ρ - 2 * m + q₀
    Δ̂ = ρ - 2 * m + q₁ + Δ # here we can provide the original ρ and Δ
    return (q₀ - q₁) / Δ̂
end

"""
this function returns the scaling of the variance such that the MSE of the ensemble is equal 
to the variance
"""
function get_scaling_matched_variance_error()::Float64
    error("not done yet")
end

# 

function experimental_get_scaling_gaussian_nll(predictions::AbstractMatrix, y_test::AbstractVector; scaling_min = 0.001, scaling_max = 1000.0)::Float64
    # predictions is a K × n matrix
    function gaussian_nll_function(scaling::Real)::Real
        means     = mean(predictions, dims=1) # vector 
        variances = var(predictions, dims=1) # vector
        scaled_variances = variances ./ scaling

        return mean(  (y_test - means) ./ scaled_variances + log.(scaled_variances) )
    end

    # find the scaling that minimizes the gaussian NLL
    res = Optim.optimize(gaussian_nll_function, [scaling_min, scaling_max])
    return Optim.minimizer(res)
end

function experimental_get_scaling_matched_variance_error()::Float64
    error("not done yet")
end
