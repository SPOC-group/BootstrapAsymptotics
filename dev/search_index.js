var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [BootstrapAsymptotics]","category":"page"},{"location":"api/#BootstrapAsymptotics.BootstrapAsymptotics","page":"API reference","title":"BootstrapAsymptotics.BootstrapAsymptotics","text":"BootstrapAsymptotics\n\nState evolution and simulation for bootstrap-related methods applied to linear and logistic regression.\n\nExports\n\nBayesOpt\nERM\nFullResampling\nLabelResampling\nLogistic\nOverlaps\nPairBootstrap\nResidualBootstrap\nRidge\nSubsampling\nbias_state_evolution\nbias_variance_empirical\nbias_variance_true\nfit\ngamp\nsample_all\nsample_data\nsample_labels\nsample_weights\nstate_evolution\nstate_evolution_BayesOpt\nvariance_state_evolution\n\n\n\n\n\n","category":"module"},{"location":"api/#BootstrapAsymptotics.BayesOpt","page":"API reference","title":"BootstrapAsymptotics.BayesOpt","text":"struct BayesOpt <: BootstrapAsymptotics.Algorithm\n\nBayes optimal estimation algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.ERM","page":"API reference","title":"BootstrapAsymptotics.ERM","text":"struct ERM <: BootstrapAsymptotics.Algorithm\n\nEmpirical Risk Minimization algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.FullResampling","page":"API reference","title":"BootstrapAsymptotics.FullResampling","text":"struct FullResampling <: BootstrapAsymptotics.Algorithm\n\nFull resampling algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.LabelResampling","page":"API reference","title":"BootstrapAsymptotics.LabelResampling","text":"struct LabelResampling <: BootstrapAsymptotics.Algorithm\n\nLabel resampling algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.Logistic","page":"API reference","title":"BootstrapAsymptotics.Logistic","text":"struct Logistic <: BootstrapAsymptotics.Problem\n\nLogistic regression problem with ridge penalty.\n\nFields\n\nα::Float64: ratio of population over dimension n/d\nλ::Float64: regularization strength\nρ::Float64: teacher weight\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.Overlaps","page":"API reference","title":"BootstrapAsymptotics.Overlaps","text":"struct Overlaps{hat, T1<:(AbstractVector), T2<:(AbstractMatrix), T3<:(AbstractMatrix)}\n\nOverlap or hat overlap storage for state evolution between two algorithms.\n\nFields\n\nm::AbstractVector\nQ::AbstractMatrix\nV::AbstractMatrix\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.PairBootstrap","page":"API reference","title":"BootstrapAsymptotics.PairBootstrap","text":"struct PairBootstrap <: BootstrapAsymptotics.Algorithm\n\nStandard (pair) bootstrap algorithm.\n\nFields\n\np_max::Int64: maximum weight for state evolution\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.ResidualBootstrap","page":"API reference","title":"BootstrapAsymptotics.ResidualBootstrap","text":"struct ResidualBootstrap <: BootstrapAsymptotics.Algorithm\n\nResidual bootstrap algorithm, aka ERM + label resampling.\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.Ridge","page":"API reference","title":"BootstrapAsymptotics.Ridge","text":"struct Ridge <: BootstrapAsymptotics.Problem\n\nLeast squares regression problem with ridge penalty.\n\nFields\n\nα::Float64: ratio of population over dimension n/d\nΔ::Float64: Gaussian noise variance\nλ::Float64: regularization strength\nρ::Float64: teacher weight\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.Subsampling","page":"API reference","title":"BootstrapAsymptotics.Subsampling","text":"struct Subsampling <: BootstrapAsymptotics.Algorithm\n\nSubsampling algorithm.\n\nFields\n\nr::Float64: subsampling fraction\n\n\n\n\n\n","category":"type"},{"location":"api/#BootstrapAsymptotics.bias_state_evolution-Tuple{BootstrapAsymptotics.Problem, BootstrapAsymptotics.Algorithm}","page":"API reference","title":"BootstrapAsymptotics.bias_state_evolution","text":"bias_state_evolution(\n    problem,\n    algo;\n    check_convergence,\n    kwargs...\n)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.bias_variance_empirical-Tuple{Random.AbstractRNG, BootstrapAsymptotics.Problem, BootstrapAsymptotics.Algorithm}","page":"API reference","title":"BootstrapAsymptotics.bias_variance_empirical","text":"bias_variance_empirical(rng, problem, algo; n, K)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.bias_variance_empirical-Tuple{Random.AbstractRNG, BootstrapAsymptotics.Problem, ResidualBootstrap}","page":"API reference","title":"BootstrapAsymptotics.bias_variance_empirical","text":"bias_variance_empirical(rng, problem, algo; n, K)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.bias_variance_empirical-Tuple{Random.AbstractRNG, Logistic, BayesOpt}","page":"API reference","title":"BootstrapAsymptotics.bias_variance_empirical","text":"bias_variance_empirical(rng, problem, algo; n, K)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.bias_variance_empirical-Tuple{Random.AbstractRNG, Ridge, BayesOpt}","page":"API reference","title":"BootstrapAsymptotics.bias_variance_empirical","text":"bias_variance_empirical(rng, problem, algo; n, K)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.bias_variance_true-Tuple{Random.AbstractRNG, BootstrapAsymptotics.Problem}","page":"API reference","title":"BootstrapAsymptotics.bias_variance_true","text":"bias_variance_true(rng, problem; n, K, conditional)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.gamp-Tuple{Logistic, AbstractMatrix, AbstractVector}","page":"API reference","title":"BootstrapAsymptotics.gamp","text":"gamp(problem, X, y; max_iter, rtol)\n\n\nRun generalized approximate message passing on logistic regression to recover the Bayes optimal estimator.\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.sample_all-Tuple{Random.AbstractRNG, BootstrapAsymptotics.Problem, Integer}","page":"API reference","title":"BootstrapAsymptotics.sample_all","text":"sample_all(rng, problem, n)\n\n\nSample X, w and y all at once for a given problem with population size n.\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.sample_data-Tuple{Random.AbstractRNG, BootstrapAsymptotics.Problem, Integer}","page":"API reference","title":"BootstrapAsymptotics.sample_data","text":"sample_data(rng, problem, n)\n\n\nSample the data matrix X for a given problem with population size n.\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.sample_labels-Tuple{Random.AbstractRNG, Logistic, AbstractMatrix, AbstractVector}","page":"API reference","title":"BootstrapAsymptotics.sample_labels","text":"sample_labels(rng, , X, w)\n\n\nSample the labels vector y for a given problem from the features X and weights w.\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.sample_weights-Tuple{Random.AbstractRNG, BootstrapAsymptotics.Problem, Integer}","page":"API reference","title":"BootstrapAsymptotics.sample_weights","text":"sample_weights(rng, problem, n)\n\n\nSample the weights vector w for a given problem with population size n (from which the dimension is deduced).\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.state_evolution-Tuple{BootstrapAsymptotics.Problem, BootstrapAsymptotics.Algorithm, BootstrapAsymptotics.Algorithm}","page":"API reference","title":"BootstrapAsymptotics.state_evolution","text":"state_evolution(\n    problem,\n    algo1,\n    algo2;\n    rtol,\n    max_iteration,\n    show_progress\n)\n\n\nPeform state evolution on a problem for the couple (algorithm1, algorithm2), by creating and then iteratively updating overlaps and hat overlaps.\n\nKeyword arguments\n\nrtol: relative tolerance used at every step of the procedure, especially to check overlap convergence\nmax_iteration: maximum number of overlap updates\nshow_progress: whether to display a progress bar\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.state_evolution_BayesOpt-Tuple{Logistic}","page":"API reference","title":"BootstrapAsymptotics.state_evolution_BayesOpt","text":"state_evolution_BayesOpt(problem; rtol, max_iteration)\n\n\nSpecial case of state evolution for the Bayes optimal estimator.\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.variance_state_evolution-Tuple{BootstrapAsymptotics.Problem, BootstrapAsymptotics.Algorithm}","page":"API reference","title":"BootstrapAsymptotics.variance_state_evolution","text":"variance_state_evolution(\n    problem,\n    algo;\n    check_convergence,\n    kwargs...\n)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#BootstrapAsymptotics.variance_state_evolution-Tuple{BootstrapAsymptotics.Problem, ResidualBootstrap}","page":"API reference","title":"BootstrapAsymptotics.variance_state_evolution","text":"variance_state_evolution(\n    problem,\n    algo;\n    check_convergence,\n    kwargs...\n)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#StatsAPI.fit","page":"API reference","title":"StatsAPI.fit","text":"fit([rng], problem, algorithm, X, y, [w_star])\n\nFit an algorithm to data X, y generated by problem, with randomness source rng.\n\n\n\n\n\n","category":"function"},{"location":"#BootstrapAsymptotics","page":"Home","title":"BootstrapAsymptotics","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Code Style: Blue)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Code for the paper","category":"page"},{"location":"","page":"Home","title":"Home","text":"Analysis of Bootstrap and Subsampling in High-dimensional Regularized RegressionClarté et al. (2024), arXiv:2402.13622","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Open a Julia console and run:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg\n\njulia> Pkg.add(url=\"https://github.com/SPOC-group/BootstrapAsymptotics\")","category":"page"},{"location":"#Using-the-package","page":"Home","title":"Using the package","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> using BootstrapAsymptotics\n\njulia> (; overlaps) = state_evolution(Ridge(λ=0.1), PairBootstrap(), PairBootstrap());\n\njulia> Matrix(overlaps.Q)\n2×2 Matrix{Float64}:\n 1.35642   0.799808\n 0.799808  1.35642","category":"page"},{"location":"#Reproducing-the-plots","page":"Home","title":"Reproducing the plots","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Open the Jupyter notebooks in the experiments folder, select the Julia environment defined in the same folder, and run.","category":"page"},{"location":"#Computing-the-large-\\alpha-rates","page":"Home","title":"Computing the large alpha rates","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Open the Mathematica notebook experiments/large_alpha_rates.nb","category":"page"},{"location":"#Notations","page":"Home","title":"Notations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"n: number of samples\nd: dimension\nα: sampling ratio n/d\nΔ: noise variance\nλ: regularization parameter\nm, Q, V: overlaps","category":"page"}]
}
