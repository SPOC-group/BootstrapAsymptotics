using BootstrapAsymptotics
using StableRNGs

rng = StableRNG(0)

n = 100
problem = Ridge()
(; X, w, y) = sample_all(rng, problem, n)

w_erm = fit(rng, problem, ERM(), X, y)
w_pair = fit(rng, problem, PairBootstrap(), X, y)
w_sub = fit(rng, problem, SubsamplingBootstrap(; r=0.1), X, y)
w_full = fit(rng, problem, FullResampling(), X, y, w)
w_label = fit(rng, problem, LabelResampling(), X, y, w)
