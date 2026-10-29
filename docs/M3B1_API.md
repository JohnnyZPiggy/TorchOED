# M3-B1 simultaneous two-swap scoring (Issue #11)

EXPLORE API/schema frozen before VALIDATE:
`score_two_swaps(F, ids, *, tile_size=64, max_elements=1_000_000)`.
This scores **all legal unordered outgoing position pairs and incoming candidate
ID pairs**, without applying swaps, choosing a winner or running a search.
Only unbatched finite CPU float64 model features F(m,p), CPU int64 unique IDs(n),
m>=n>=p>=1 and an initial Gram accepted by M2-A numerical SPD policy. There is
no k/replication/initializer/search parameter. No RNG consumption or mutation of
inputs. No discrete-selection/autograd guarantee; scoring runs under no_grad.

## Stable schema and enumeration

Let q=m-n and C=binom(n,2)*binom(q,2). Rows are ordered lexicographically by
outgoing **positions** i1<i2 in the supplied ID sequence, then by unselected
candidate IDs j1<j2. Do not sort or deduplicate IDs by feature equality. Each row
means simultaneous assignment trial[positions]=incoming_ids, preserving other
positions. Distinct IDs with equal features remain separate candidates.

The result dict contains:

- initial_ids (cloned int64[n]), initial_objective (Python float).
- outgoing_positions and incoming_ids, int64[C,2]. Outgoing candidate IDs are
  initial_ids[outgoing_positions], distinct from outgoing positions.
- ratios: float64[C] raw rank-four determinant ratios, including nonfinite/zero
  or negative diagnostics; never clipped.
- gains: float64[C] **direct verified** D_trial-D_initial only for VALID records,
  otherwise NaN as explicit undefined sentinel, never successful NaN output.
- direct_scores: float64[C] wherever M2-A accepts a trial, even if ratio check
  is inconclusive; otherwise NaN. No gain inferred from ratio sign alone.
- tolerances and log_ratio_errors: float64[C], NaN where checks are undefined.
- status: int8[C], status_names {0:VALID,1:INVALID,2:NUMERICAL_INCONCLUSIVE};
  reasons: list of C diagnostic strings. counts summarizes each status.
- evaluated_pairs, effective_tile_size, workspace_element_estimate;
  certificate='NOT_APPLICABLE'. No winner, selected-endpoint, history, stop or
  local/global certificate API. B11 search/monotonicity is NOT_APPLICABLE.

If n<2 or q<2, return empty arrays with the same shapes/types, count0, tile0
after validating the initial design. Invalid inputs/initial SPD/budget raise
TypeError/ValueError. Invalid individual trials are represented by status.

## Mathematics, verification and finite precision

With original M=LLᵀ, W=[Oᵀ,Nᵀ] and J=diag(-1,-1,+1,+1), solve Z=L⁻¹W using
torch.linalg.solve_triangular; Gamma2=det(I4+J ZᵀZ), native batched 4x4 det.
No explicit inverse or deletion-only factorization; n=p is allowed. For every
pair independently rebuild full selected-feature Gram and call merged
exact_information/d_optimality. No upstream code copied, PyDOE backend, ridge
or Gamma clipping. Determinant-ratio sign alone never establishes SPD.

M2-A rejection plus an exactly zero numerical QR diagonal gives INVALID with
reason NUMERICAL_QR_ZERO; this is a numerical rank-loss diagnostic, **not an
algebraic proof**. A nonzero QR diagonal gives NUMERICAL_INCONCLUSIVE, preserving
near-singular mathematical SPD rather than mislabeling it rank deficient.
Other M2 nonfinite/overflow cases likewise do not receive a valid score.

On directly SPD trials require finite Gamma2>0 and agreement between logGamma2
and direct D difference within
`256*eps64*p*max(1,|D_initial|,|D_trial|)*max(1,cond2(M),cond2(M_trial))`.
This condition-aware **heuristic** is not a rigorous error interval. Ratios
nonpositive/nonfinite on SPD trials or disagreement remain inconclusive. Gains
can be negative, zero or positive: scoring does not accept/reject improvements.

## Bounds and scoped oracle evidence

Compute C with Python integers before neighborhood/output allocation. The
budget counts retained tensor output elements 10C+n, fixed major workspace
2mp+6np+10p²+4m+2n+64, and per tile pair16p+128. Reduce width to fit the remaining
budget; reject if even one pair cannot fit. F/selected/Gram shapes also use
M2's preallocation guards. Neighbor index tuples are generated in bounded tiles;
no full Cartesian pair list is created. Retained output remains O(C), explicitly
bounded. These estimates are not a peak-memory guarantee for native linalg
workspace, Python reason/tuple objects, allocator caches or result serialization.

Tests use independent stdlib Fraction/Bareiss Gram determinants and exact
subset/pair enumeration, not production gain output as expected values.
The required count150 is n6/m11 **total**, incoming5, not Mentor's m17/incoming11
count825. B1/B2 four-point trap, B4 endpoint identity, B6 k1 mathematical lemma,
B7 all150 pairs, B8 saturated, B9 distinct duplicate features, B10 ordering and
B12 boundaries are scoped checks. B3/B5 Modified Fedorov/excursion code NOT_RUN;
B11 search stopping and local/global certificates NOT_APPLICABLE. Comparisons
use exact determinants/IDs and declared float64 abs/rel1e-9 on well-conditioned
fixtures. No float32/batch/GPU/search/benchmark/upstream parity claim.
