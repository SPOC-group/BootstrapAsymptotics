using BootstrapAsymptotics
using StableRNGs

rng = StableRNG(0)

n = 100
for problem in (Logistic(), Ridge())
    (; X, w, y) = sample_all(rng, problem, n)
end
