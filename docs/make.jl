using BootstrapAsymptotics
using Documenter

makedocs(;
    modules=[BootstrapAsymptotics],
    authors="SPOC and IdePHICS labs",
    sitename="BootstrapAsymptotics.jl",
    format=Documenter.HTML(),
    pages=["Home" => "index.md"],
)
