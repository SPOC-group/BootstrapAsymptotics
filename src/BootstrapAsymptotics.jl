module BootstrapAsymptotics

using HCubature: hcubature
using IntervalArithmetic: interval
using LinearAlgebra: Diagonal, Symmetric, I, dot, norm
using LogExpFunctions: log1pexp
using NLSolvers: NLSolvers
using QuadGK: quadgk
using Random: AbstractRNG
using StatsFuns: normpdf, poispdf, logistic
using StaticArrays: SVector, SMatrix

include("utils/problems.jl")
include("utils/algos.jl")
include("utils/overlaps.jl")
include("utils/state_evolution.jl")

include("ridge/simulation.jl")
include("ridge/state_evolution.jl")

include("logistic/simulation.jl")
include("logistic/state_evolution_single.jl")
include("logistic/state_evolution.jl")

export Overlaps
export Ridge, Logistic
export PairBootstrap, ResidualBootstrap, SubsamplingBootstrap
export ERM, LabelResampling, FullResampling
export state_evolution

end
