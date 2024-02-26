using Documenter
using BootstrapAsymptotics

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

pages = ["Home" => "index.md", "API reference" => "api.md"]

fmt = Documenter.HTML(;
    repolink="https://github.com/SPOC-group/BootstrapAsymptotics",
    canonical="https://SPOC-group.github.io/BootstrapAsymptotics",
)

makedocs(;
    modules=[BootstrapAsymptotics],
    authors="SPOC group",
    sitename="BootstrapAsymptotics",
    format=fmt,
    pages=pages,
)

deploydocs(; repo="https://github.com/SPOC-group/BootstrapAsymptotics", devbranch="main")
