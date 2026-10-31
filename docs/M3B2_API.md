# M3-B2: deterministic k=2 VND

`k2_exchange(F, ids, *, max_iter=100, tile_size=64, max_elements=1_000_000)`
is the 14th root export. Inputs follow the unchanged M3-A/M3-B1 contracts:
unbatched finite CPU float64 features, unique CPU int64 IDs, initially
numerically SPD information matrix, no replacement or ridge.

At each accepted state, factor M, solve Z = L^-1 F.T once, then form G_SS
and G_SN. Phase 1 scores all one-swaps with the dispersion identity;
Phase 2 streams gather4 determinants of I + diag(-1,-1,1,1) G_block.
G_NN is tiled by incoming IDs; the full two-swap neighborhood is never stored.
Each phase keeps only eight candidates, ordered by descending estimated ratio
and lexicographic row order within the tolerance band of the global maximum
(per the Scientist's #43 clarification). Near-ties trigger a second bounded scan,
so tile order cannot change the band anchor. Only ratios above 1 plus the
condition-aware ratio tolerance are eligible. Verify in that order with a
fresh Gram + d_optimality, accepting the first direct gain above _tolerance.
An accepted move restarts Phase 1. Both public reference scorers remain unchanged.
An empty N2 reports NO_K2_NEIGHBORHOOD after Phase 1 unless no legal swap exists.
max_iter bounds accepted moves. max_elements bounds state/tile workspace,
not the number of neighbors; insufficient space for one tile raises ValueError.

The returned dict has `ids`, `initial_objective`, `final_objective`,
`objective_history`, `swap_history`, `accepted_swaps`, `scanned_pairs`
(`k1`, `k2`), `verification_attempts`, `stop_reason`,
`full_neighborhood_checked`, `certificate` and `efficiency_gap_bound`.
Each move records `k`, `outgoing_positions`, `outgoing_ids`, `incoming_ids`,
`gamma`, direct `gain` and `tolerance`. Counts aggregate all phases, including repeat scoring scans for near-ties;
verification_attempts counts only direct trial checks.

Stop reasons: `LOCAL_2_NUMERICAL`, `NUMERICAL_INCONCLUSIVE`, `MAX_ITER`,
`NO_K2_NEIGHBORHOOD` (n<2 or m-n<2 after Phase 1), `NO_LEGAL_SWAP`.
If a phase has eligible ratios but all retained candidates fail direct verification,
stop NUMERICAL_INCONCLUSIVE. If neither complete scan has an eligible ratio,
stop LOCAL_2_NUMERICAL. The tolerance policy is heuristic, not an interval proof.
Only completed final N1/N2 scans yield `certificate="LOCAL_2_NUMERICAL"`
and `full_neighborhood_checked=True`; every other stop is `NOT_CERTIFIED`.
This is numerical 1-/2-locality, never exact or global optimality.
The gate's exact reference retains 60 non-global 2-local endpoints.

efficiency_gap_bound = p * log(n * max_j G_jj / p), evaluated at the final
state, bounds logdet(optimum) - final_objective over all n-subsets in exact
arithmetic. It is an upper bound on the gap, never achieved optimality;
its floating-point evaluation carries numerical error.
