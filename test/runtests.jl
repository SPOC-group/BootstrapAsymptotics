using Aqua
using BootstrapAsymptotics
using JuliaFormatter
using Test

@testset verbose = true "BootstrapAsymptotics" begin
    @testset "Ridge" begin
        include("ridge.jl")
    end
end
