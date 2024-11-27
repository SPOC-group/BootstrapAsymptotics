"""
    BootstrapAsymptotics

State evolution and simulation for bootstrap-related methods applied to linear and logistic regression.

# Exports

$(EXPORTS)
"""
module BootstrapAsymptotics

using Base.Iterators: product
using DocStringExtensions
using HCubature: hcubature
using LinearAlgebra: Diagonal, Symmetric, I, dot, norm
using LogExpFunctions: log1pexp
using MLJLinearModels: MLJLinearModels
using NLSolvers: NLSolvers
using Optim
using ProgressMeter: Progress, next!
using QuadGK: quadgk
using Random: AbstractRNG
using StableRNGs: StableRNG
using Statistics: mean
using StatsBase: sample
using StatsAPI: StatsAPI, fit
using StatsFuns: normpdf, poispdf, logistic
using StaticArrays: SVector, SMatrix

include("utils.jl")
include("problems.jl")
include("algos.jl")

include("fit.jl")
include("sample.jl")

include("overlaps.jl")
include("state_evolution.jl")
include("state_evolution_ridge.jl")
include("state_evolution_logistic.jl")
include("state_evolution_bayesopt.jl")
include("state_evolution_kernel.jl")

include("bias_variance.jl")
include("gamp.jl")

include("regularization.jl")
include("scaling.jl")


export Overlaps
export Ridge, Logistic
export PairBootstrap, Subsampling, ResidualBootstrap, NoResampling
export ERM, BayesOpt, LabelResampling, FullResampling
export sample_data, sample_weights, sample_labels, sample_all
export fit
export gamp
export state_evolution, state_evolution_BayesOpt
export bias_variance_true,
    bias_variance_empirical, variance_state_evolution, bias_state_evolution, overlaps_empirical
export get_additional_noise_from_kappas

end
