module BootstrapAsymptotics

using IntervalArithmetic: interval
using LinearAlgebra: Diagonal, Symmetric, I, norm
using QuadGK: quadgk
using Random: AbstractRNG
using StatsFuns: normpdf, poispdf, logistic
using StaticArrays: SVector, SMatrix

include("utils/problems.jl")
include("utils/algos.jl")
include("utils/overlaps.jl")

include("ridge/simulation.jl")
include("ridge/state_evolution.jl")

include("logistic/simulation.jl")

export Ridge, Logistic
export PairBootstrap, ResidualBootstrap, LabelResampling, FullResampling
export state_evolution

end
