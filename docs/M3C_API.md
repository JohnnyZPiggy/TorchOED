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
