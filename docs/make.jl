using BootstrapAsymptotics
using Documenter

makedocs(;
    modules=[BootstrapAsymptotics],
    authors="Anonymous authors",
    sitename="BootstrapAsymptotics.jl",
    format=Documenter.HTML(),
    pages=["Home" => "index.md"],
)
