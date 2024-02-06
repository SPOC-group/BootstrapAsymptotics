using Aqua
using BootstrapAsymptotics
using JuliaFormatter
using Pkg
using Test

@testset verbose = true "BootstrapAsymptotics" begin
    @testset "Code quality" begin
        Aqua.test_all(BootstrapAsymptotics; ambiguities=false)
    end
    @testset "State evolution" begin
        include("state_evolution.jl")
    end
    @testset "Variance" begin
        include("variance.jl")
    end
    @testset "Bayes optimal" begin
        include("bayesopt.jl")
    end
    # @testset "Comparison" begin
    #     include("../comparison/ridge.jl")
    #     include("../comparison/logistic.jl")
    #     Pkg.activate(dirname(@__DIR__))
    # end
end
