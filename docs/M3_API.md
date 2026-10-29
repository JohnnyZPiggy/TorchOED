# M3-A Fedorov exchange: CPU numerical search

`fedorov_exchange(F, ids, *, max_iter=100, tile_size=64,
allow_repeats=False, max_elements=1_000_000)` is the only M3-A public addition.
F is **model features**, not physical factors. Accepts only finite CPU float64
F(m,p), CPU int64 ids(n), m>=n>=p>=1, unique in-range IDs and an initial Gram
accepted by merged M2-A's numerical SPD policy. Bad initial designs raise;
there is no initializer. allow_repeats=True raises NotImplementedError. Inputs
are unchanged; selected-ID order is preserved except the replaced position.
The routine runs under no_grad, consumes no RNG and makes no differentiation
claim for the discrete selection. No batches, float32, CUDA, M3-B, multistart,
Modified Fedorov, DETMAX or k-swap is implemented. Version stays0.0.1.

## Numerical search and best-move rule

Every sweep considers **all** n(m-n) legal pairs (outgoing position,incoming ID).
The pool excludes exact selected IDs; different IDs with identical features
remain legal. With current M=LLᵀ, native triangular solves give R=L⁻¹Xᵀ and
T=L⁻¹Bᵀ; tiled Γ=(1-u)(1+v)+c². There is no inverse, delete-only factorization,
Gamma clipping, hidden ridge, or top-K screening. Every neighbor is independently
reconstructed through full selected-feature Gram and merged d_optimality.

For a verified SPD trial define heuristic gain tolerance
`g=128*eps64*p*max(1,abs(D_old),abs(D_trial))*max(1,cond2(M),cond2(M_trial))`.
Direct Δ must exceed g and abs(logΓ-Δ)<=g, with finite Γ>0, before eligibility.
This is an explicit conservative **heuristic**, not a proven error interval.
Current/trial condition numbers are measured on the small Gram matrices.

First find the globally largest **direct verified** score among eligible moves.
Anchor the tie tolerance to that best move's g; recheck a lexicographic prefix
of the legal neighborhood, choosing the first eligible score within that
tolerance of the maximum. Outgoing position then incoming ID determine order.
This avoids nontransitive streaming tolerance ties and keeps the choice
independent of tile traversal. Strict accepted objective history is retained.

A failed M2 SPD/finite check is followed by full selected-feature QR. An exactly
zero numerical R diagonal is recorded as `qr_zero` (a numerical rank-loss
diagnostic, **not** a rigorous algebraic certificate). A nonzero QR diagonal with
M2 rejection remains unresolved, including mathematically SPD near-singular
cases: `M2_REJECT_QR_NONZERO`. Nonpositive/nonfinite Γ or ratio/direct disagreement
on a directly SPD trial is unresolved. Any such unresolved neighbor stops the
whole sweep with NUMERICAL_INCONCLUSIVE, even if another move looks attractive.

Tiny positive Δ<=g is recorded separately as marginal, with heuristic upper
gain Δ+g. If a verified best move's lower gain Δ_best-g_best exceeds every
marginal upper gain, accept the best eligible/tied move; otherwise stop
NUMERICAL_INCONCLUSIVE. This avoids an irrelevant roundoff tie blocking a clear
improvement while retaining explicit uncertainty at stopping. The inequalities
are numerical policy only; they do not prove universal best-exchange bounds.

## Results, stopping and resources

Result dict includes ids Tensor; Python initial/final objectives;
objective_history; swap_history (outgoing position/ID, incoming ID, gain, Γ,
direct score, acceptance/tie tolerance, best verified score); scans (counts,
condition, max ratio discrepancy/tolerance, first unresolved reason);
accepted_swaps/evaluated_candidates/direct_trial_evaluations, stop_reason,
full_neighborhood_checked, certificate and effective tile/workspace estimate.

- LOCAL_1_SWAP_NUMERICAL: complete final-state sweep, no accepted improvement
  under this policy and no unresolved/marginal positive gain. certificate is
  NUMERICAL_ONLY, never LOCAL_1_SWAP_CERTIFIED or a global optimality claim.
- NUMERICAL_INCONCLUSIVE: complete sweep contains unresolved candidates or
  competitive marginal gains; no move from that sweep accepted, certificate NONE.
- MAX_ITER: accepted-swap limit reached before scanning final-state neighborhood,
  including max_iter=0. certificate NONE, full_neighborhood_checked=False.
- NO_LEGAL_SWAP: n=m, no scan needed. certificate NONE. This label takes precedence
  over max_iter when no legal neighbors exist.

evaluated_candidates counts all legal pairs once per completed primary sweep;
direct_trial_evaluations also counts the prefix tie rechecks, excluding the
initial score. No complete n×m allocation or retained per-pair history. The
declared max_elements bounds input/selected/Gram tensors via M2 and a conservative
major-tensor scratch estimate `fixed=4np+8p²+2m+4n`, per tile column `6n+4p+8`;
effective tile width=min(requested,floor((budget-fixed)/per_column)). Reject if
one column cannot fit. `8*(max_iter+1)<=budget` additionally bounds history length.
These count estimates are **not** measured peak memory or bounds on backend
SVD/QR workspace, Python objects or allocator caching. All external validation
commands have finite timeouts; callers must select their own practical budget.

## Independent evidence

F1–F12 in tests/test_m3a.py follow M3_FEDOROV_EXCHANGE_GATE, not M1's older F labels.
tests/oracle_exchange.py uses only stdlib Fraction/itertools, independent literal
Gram construction and elimination; all distinct-ID subsets for m<=8 and full
1-swap neighborhoods are enumerable. Integer/rational determinants use equality;
Γ/direct score comparisons on small well-conditioned cases use abs/rel1e-9.
Mentor's four-candidate trap has local determinant1 and global2.6244: all six
subsets are checked. This is a required local-vs-global negative control.

Native torch primitives and merged M1/M2 APIs are composed without copying
PyDOE source or calling its wheel. Runtime dependency remains torch only.
See reports/m3a/ENGINEER_HANDOFF.md for actual source/commands/errors/failures.
No performance, other-platform, rigorous interval or GPU capability is claimed.
