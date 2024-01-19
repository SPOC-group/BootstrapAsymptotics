using BootstrapAsymptotics

overlaps, hatoverlaps = state_evolution(Ridge(); weight_dist=indep_poisson, α=1.0, λ=1e-4, σ²=1.0);
overlaps.m
overlaps.Q
overlaps.V
hatoverlaps.m̂
hatoverlaps.Q̂
hatoverlaps.V̂
