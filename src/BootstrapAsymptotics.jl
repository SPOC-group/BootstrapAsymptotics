module BootstrapAsymptotics

using HCubature
using IntervalArithmetic: interval
using LinearAlgebra: Diagonal, Symmetric, I, norm
using NLSolvers: NLSolvers
using QuadGK: quadgk
using Random: AbstractRNG
using StatsFuns: normpdf, poispdf, logistic
using StaticArrays: SVector, SMatrix

include("utils/problems.jl")
include("utils/algos.jl")
include("utils/losses.jl")
include("utils/sample.jl")
include("utils/overlaps.jl")
include("utils/state_evolution.jl")

include("ridge/simulation.jl")
include("ridge/state_evolution.jl")

include("logistic/simulation.jl")
include("logistic/state_evolution_single.jl")
include("logistic/state_evolution.jl")

export Overlaps
export Ridge, Logistic
export ERM, PairBootstrap, ResidualBootstrap, LabelResampling, FullResampling
export state_evolution

end
