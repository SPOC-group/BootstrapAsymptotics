using Aqua
using BootstrapAsymptotics
using JuliaFormatter
using Pkg
using Test

@testset verbose = true "BootstrapAsymptotics" begin
    @testset "Code quality" begin
        Aqua.test_all(BootstrapAsymptotics; ambiguities=false)
    end
    @testset "Ridge" begin
        include("../comparison/ridge.jl")
    end
    @testset "Logistic" begin
        include("../comparison/logistic.jl")
    end
    @testset "Code performance" begin
        include("types_allocations.jl")
    end
end

Pkg.activate(dirname(@__DIR__))
