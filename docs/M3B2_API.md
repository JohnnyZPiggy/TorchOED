# M3-B2: deterministic k=2 VND

`k2_exchange(F, ids, *, max_iter=100, tile_size=64, max_elements=1_000_000)`
is the 14th root export. Inputs follow the unchanged M3-A/M3-B1 contracts:
unbatched finite CPU float64 features, unique CPU int64 IDs, initially
numerically SPD information matrix, no replacement or ridge.

Phase 1 calls `fedorov_exchange` until numerical 1-locality. Phase 2 calls
`score_two_swaps`, selects the greatest VALID direct gain exceeding its own
tolerance (first row on exact ties), applies that move and returns to Phase 1.
Both scorers rebuild information/factors; no gather kernel or factor updates.
An earlier Phase-1 stop propagates unchanged; it does not certify locality.
`max_iter` bounds total accepted one- and two-swaps, not outer VND cycles.
Workspace/output limits are enforced by the reused APIs; no new limit policy.

The returned dict has `ids`, `initial_objective`, `final_objective`,
`objective_history`, `swap_history`, `accepted_swaps`, `scanned_pairs`
(`k1`, `k2`), `verification_attempts`, `stop_reason`,
`full_neighborhood_checked` and `certificate`.
Each move records `k`, `outgoing_positions`, `outgoing_ids`, `incoming_ids`,
`gamma`, direct `gain` and `tolerance`. Counts aggregate all phases, including
the one-swap scorer's direct tie rechecks in verification attempts.

Stop reasons: `LOCAL_2_NUMERICAL`, `NUMERICAL_INCONCLUSIVE`, `MAX_ITER`,
`NO_K2_NEIGHBORHOOD` (n<2 or m-n<2 after Phase 1), `NO_LEGAL_SWAP`.
Unresolved two-swap rows prevent certification when no verified move remains.
Only completed final N1/N2 scans yield `certificate="LOCAL_2_NUMERICAL"`
and `full_neighborhood_checked=True`; every other stop is `NOT_CERTIFIED`.
This is numerical 1-/2-locality, never exact or global optimality.
The gate's exact reference retains 60 non-global 2-local endpoints.
