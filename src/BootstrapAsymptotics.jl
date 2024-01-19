module BootstrapAsymptotics

using LinearAlgebra: Diagonal, I, norm
using QuadGK: quadgk
using Random: AbstractRNG
using StatsFuns: normpdf, poispdf, logistic
using StaticArrays: SVector, SMatrix

include("utils/math.jl")
include("utils/types.jl")

include("ridge/simulation.jl")
include("ridge/state_evolution.jl")

include("logistic/simulation.jl")

export Ridge, Logistic
export indep_poisson
export state_evolution

end
