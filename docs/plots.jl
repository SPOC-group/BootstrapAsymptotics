using Base.Threads
using BootstrapAsymptotics
using Colors
using Plots
using ProgressMeter
using Statistics
using StableRNGs

#=
using PGFPlotsX
pgfplotsx()  # for LaTeX-compatible plots
=#

nthreads()

rng = StableRNG(10)

d = 200
K = 20
λ = 0.1
setting = :logistic

α_vals = 10 .^ (-1:0.2:1)
algo_vals = [
    PairBootstrap(; p_max=5), #
    SubsamplingBootstrap(0.8), #
    FullResampling(), #
    LabelResampling(), #
]
colors = distinguishable_colors(
    length(algo_vals), [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true
);

vars_emp = Dict(algo => fill(NaN, length(α_vals)) for algo in algo_vals)
vars_se = Dict(algo => fill(NaN, length(α_vals)) for algo in algo_vals)

for algo in algo_vals
    algo isa LabelResampling || continue
    @threads for i in eachindex(α_vals)
        α = α_vals[i]
        @info "$algo - α=$α"
        problem = setting == :ridge ? Ridge(; λ, α) : Logistic(; λ, α)
        _, var_emp = bias_variance_empirical(rng, problem, algo; n=ceil(Int, α * d), K)
        var_se = variance_state_evolution(
            problem,
            algo;
            check_convergence=false,
            show_progress=false,
            rtol=1e-4,
            max_iteration=100,
        )
        vars_emp[algo][i] = var_emp
        vars_se[algo][i] = var_se
    end
end

pl = plot(; title=setting == :ridge ? "Ridge(λ=$λ)" : "Logistic(λ=$λ)")
for (i, algo) in enumerate(algo_vals)
    scatter!(pl, α_vals, vars_emp[algo]; label="$algo", color=colors[i])
    plot!(pl, α_vals, vars_se[algo]; label=nothing, lw=2, color=colors[i])
end
plot!(pl; xlabel="α", ylabel="variance", xscale=:log, yscale=:log, legend=:bottom)
pl
# savefig(pl, joinpath(@__DIR__, "plots", "plot.pdf"))
# savefig(pl, joinpath(@__DIR__, "plots", "plot.tex"))
