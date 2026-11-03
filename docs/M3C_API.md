# M3-C: seeded D-optimal design restarts

`d_optimal_design(F, n, *, restarts=10, generator=None, max_iter=100,
tile_size=64, max_elements=1_000_000)` is the 15th root export.
Use finite unbatched CPU float64 model features F(m,p), with integer m>=n>=p.
Search and numerical SPD policies come from the existing M2/M3 APIs.

Each restart draws `torch.randperm(m, generator=generator)[:n]`, sorted.
Rejected numerical-SPD starts are redrawn; 100*restarts failed draws in total
raise ValueError, suggesting the candidate set may be rank-deficient for n.
Each feasible start runs `k2_exchange` with the supplied search/resource limits.
The best final objective wins; an exact tie retains the first restart.

Returns `ids`, `objective`, `stop_reason`, `certificate` of the best run,
plus `restart_objectives`, zero-based `best_restart`, and initial ID lists
`starts`, both lists in restart order. This is a heuristic search, not a
guarantee of global optimality; the chosen run's certificate is unchanged.
Recreate a CPU `torch.Generator().manual_seed(seed)` for repeatable calls;
the supplied generator is advanced. With None, the global RNG is used.
There is no parallel restart execution, GPU support or smarter initializer.

Also returns the best run's `efficiency_gap_bound` without recomputing it,
and `d_efficiency_lower_bound = exp(-efficiency_gap_bound / p)`. This is a
certified lower bound on D-efficiency against the true exact n-subset optimum,
never achieved efficiency or optimality. The theorem holds in exact arithmetic;
the returned float64 values carry numerical error (no interval arithmetic).

## Generic criteria
`optimal_design(F, n, *, criterion="D", restarts=10, generator=None,
max_iter=100, max_elements=1_000_000)` accepts CPU float64 features.
D delegates unchanged to `d_optimal_design`, retaining its fields and adding
`criterion` and `efficiency_lower_bound = d_efficiency_lower_bound`.
A maximizes `-trace(M⁻¹)`; it depends on the supplied feature scaling/coding.
I maximizes `-trace(B M⁻¹)`, with uniform candidate moment `B=FᵀF/m`.
A/I use seeded feasible restarts and directly verified best-improvement 1-swaps.
Their certified floor is `trace(B M⁻¹)/(n max_j f_jᵀM⁻¹BM⁻¹f_j)` (A: B=I).
They return ids, objective, criterion, efficiency_lower_bound, stop_reason,
certificate, restart_objectives, best_restart and starts.
`LOCAL_1_NUMERICAL` certifies only numerical 1-swap locality, never global
optimality; MAX_ITER/NUMERICAL_INCONCLUSIVE return NOT_CERTIFIED.
