using Aqua
using BootstrapAsymptotics
using JuliaFormatter
using Test

@testset verbose = true "BootstrapAsymptotics" begin
    @testset "Code quality" begin
        Aqua.test_all(BootstrapAsymptotics; ambiguities=false)
    end
    @testset "Ridge" begin
        include("ridge.jl")
    end
    @testset "Logistic" begin
        include("logistic.jl")
    end
end
