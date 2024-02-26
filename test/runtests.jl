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
end
