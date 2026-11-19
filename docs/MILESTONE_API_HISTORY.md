# Historical milestone API contracts

These historical milestone API contracts are superseded by [the current API reference](../website/api.md). Historical `reports/` handoffs link the old paths, which resolve at their own commits.

## M1 CPU constructor contract

_Formerly `docs/M1_API.md`._

Issue #5 dispatches only five public callables. Version remains 0.0.1: no release
or general PyDOE compatibility is claimed. Runtime dependency remains torch only.
CPU oracles: `tests/test_m1.py`; reproducible validation: `tools/validate_m1.py`.

All scalar counts are Python ints (bool and integral floats reject). All APIs
accept `max_elements=1_000_000`, a configurable output element budget, checked
before output allocation, with an additional int64 size limit. This is not a
bound on native backend workspace or execution time. Only float32 and float64
are supported floating dtypes. Default device is CPU and floating dtype float64.
Device forwarding exists; this dispatch verifies CPU only.

| API | Frozen behavior |
|---|---|
| `full_factorial(levels, *, device=None, max_elements=...)` | Nonempty tuple/list of positive ints; int64 indices; last factor fastest. Single factors keep a column dimension. Set equivalence does not promise PyDOE row order. |
| `ff2n(k, *, device=None, dtype=torch.float64, max_elements=...)` | Positive k; signed -1/+1 corners in the same order. No implicit physical scaling. |
| `lhs(n_samples,n_factors, *, generator=None,device=None,dtype=torch.float64,mode='random',max_elements=...)` | Positive counts; one randperm and independent jitter vector per column. Only random mode. A supplied generator is consumed normally and never reseeded; global RNG is untouched. None uses native global RNG. Replay is limited to the same torch/backend. |
| `sobol(n,d, *, scramble=False,seed=None,skip=0,balance='exact',dtype=torch.float64,device=None,max_elements=...)` | Exactly n points, no padding; n=0 returns (0,d) without constructing an engine. Native CPU SobolEngine plus explicit device transfer; no CUDA-native generation claim. |
| `polynomial_features(C,degree=2,include_bias=True, *, max_elements=...)` | Floating (m,k) or (B,m,k), k>=1. Preserves dtype/device/autograd. Empty rows/batches supported. Rejects nonfinite/integer/complex input; caller must code integer indices explicitly. |

LHS repairs floating rounding across stratum boundaries with that stratum's
midpoint, ensuring exact `floor(n*x)` IDs. Counts are limited to 2^22 for float32
and 2^51 for float64 for midpoint representability; the default budget is smaller.
No maximin, centered, OA or sliced LHS is implemented. Permutation ties and
cross-library random sequences are not scientific equivalence criteria.

Sobol validates d against native MAXDIM (21201 in tested torch), skip>=0 and
skip+n<=2^MAXBIT (MAXBIT=30). Seeds are None or Python ints in [0,2^63-1]; scramble
is bool; only balance='exact' is supported. A fixed scrambling seed isolates
global RNG; seed=None with scrambling consumes native global RNG. Zero output
does not consume RNG. Direction tables and replay are torch-version specific.

Feature degree1 order: [bias, linear]. Degree2: [bias, linear, squares,
pair products i<j in lexicographic order]. Bias removal only removes the first
column. p=1+2k+k(k-1)/2 for degree2 with bias. No rank repair, ridge, information
matrix criterion, search or M2/M3 algorithm is added. Finiteness validation
requires one device synchronization on accelerators; these are untested here.

P6 reuses a SHA-256-verified immutable E0 actual-wheel output, not a fresh PyDOE
call: `tests/fixtures/e0_p1.json` records its source commit/path/hash. Actual
20x4 coordinates produce 20x15 features, rank15 and exact Gram determinant
18014398509481984. The separate D=ABC mathematical fixture has rank12/det0.
Independent scalar feature formulas and Fraction elimination supplement CPU
SVD rank. Float64 tolerance is 1e-9; float32 checks use 1e-6; gradcheck uses
eps=1e-6, atol=1e-5, rtol=1e-3. Integer IDs/strata are exact.

### Constraints and categorical factors
`effects_coding(codes, levels)` accepts int64 codes 0..levels−1 and returns
float64 columns: the first levels−1 levels are unit rows, the last is all −1.
`feasible_candidates(C, *, A=None, b=None, disallowed=())` returns a bool mask:
keep rows satisfying `C @ A.T <= b` and excluding every matching dictionary
`{column: value}` in disallowed. Comparisons are exact; add tolerance to b.
Filter with `C[mask]`; map filtered design IDs to originals using
`mask.nonzero().flatten()[ids]`. Search efficiency bounds refer to that feasible set.

## M2-A CPU information and criteria

_Formerly `docs/M2_API.md`._

Issue #7 authorizes exactly five additions, composing native torch gather/matmul,
Cholesky/solve and eigvalsh. Runtime remains torch only (no upstream code copied).
These are homoscedastic linear-model building blocks, not a nonlinear/noise-aware
Fisher engine, optimizer, efficiency percentage or general PyDOE parity claim.

All accept keyword `max_elements=1_000_000`. Input, output and selected-feature
shapes are bounded before gathering, factorizing or allocating output; dimensions
must also fit int64. The budget is not a bound on native linalg workspace.
Supported real dtypes are float32/float64, preserving input dtype/device/autograd.
Only Windows CPU on the recorded torch version has been tested.

| API | Input/output and policy |
|---|---|
| `exact_information(F,ids, *, normalized=False,allow_repeats=False,max_elements=...)` | F(m,p), ids(n) int64 → M(p,p), or matched F(B,m,p), ids(B,n) → M(B,p,p). Raw XᵀX; divide by n only explicitly. Exact IDs in [0,m), no duplicates unless allow_repeats=True. Distinct IDs at identical coordinates remain valid. |
| `weighted_information(F,w, *, max_elements=...)` | w(m) or matched w(B,m), same dtype/device as F, finite/nonnegative, sum1 within absolute 64*eps per case. Fᵀ(wF), no dense diagonal, implicit normalization or count conversion. |
| `d_optimality(M, *, max_elements=...)` | Scalar or (B,) logdet, maximize; validated SPD via cholesky_ex and numerical pivot guard, 2 sum(log diag L). |
| `a_optimality(M, *, max_elements=...)` | Scalar or (B,) -trace(M⁻¹), maximize; Cholesky solves against identity, never explicit inverse or ridge. |
| `e_optimality(M, *, max_elements=...)` | Scalar or (B,) minimum eigenvalue, maximize; symmetric numerical PSD, singular zero accepted. |

No implicit cross-batch broadcasting: B>=1, p>=1. m=0/n=0 are valid for raw
exact information with no selected events, returning zero; normalized n=0 rejects.
Empty weighted designs and empty criterion matrices/batches reject. Counts/flags
are strict types; bool/float IDs reject. Input/output nonfinite values reject.
Finite huge inputs whose arithmetic overflows also raise, never PASS with NaN/Inf.

Shape/type errors raise ValueError/TypeError. For data-dependent invalid batched
cases ValueError reports the first failing batch index (zero based); the whole
call fails, with no mixture of silently valid and invalid results.

For each M let `tau=64*p*eps*max(abs(M))` (no unit-scale floor). Require symmetry
residual max|M-Mᵀ|<=tau; within tolerance use `(M/2+Mᵀ/2)` consistently, avoiding
triangle bias. For D/A, cholesky_ex info must be zero **and every squared diagonal
of L must exceed tau**. This rejects numerical singularity/ill-conditioning even
when roundoff falsely reports Cholesky success, as observed on the exact rank12
negative control. It is a conservative, order-dependent pivot guard, not a full
condition-number certificate; some mathematically SPD but poorly scaled matrices
are deliberately rejected. No finite SPD claim is made there; no tolerance is
loosened to force a pass. E rejects eigenvalue<-tau and clips [-tau,0] roundoff to
zero; this numerical PSD policy does not claim those tiny negatives are exact PSD.
Scale=0 gives tau=0. Validation may synchronize a device; CUDA is NOT_RUN.

Repeated smallest E eigenvalues and PSD/clipping boundaries have nonunique or
nondifferentiable gradients: NOT_RUN/undefined. Float64 gradcheck on SPD simple
eigenvalue inputs perturbs free factors Q in M=QᵀQ+I, maintaining symmetry/domain;
also checks exact-information features and softmax-normalized weights.

A1–A12 numbering follows M2_IMPLEMENTATION_GATE.md (the older general contract
has different historical numbering). Tests use literal 2×2 matrices, scalar
formulas, immutable M1 E0 coordinates and independent Fraction elimination, not
production outputs as expected values. A12 actual PyDOE-derived positive is
rank15/det18014398509481984; separate D=ABC negative is rank12/det0. No new PyDOE
call. Rank thresholds: native SVD `max(m,p)*eps*sigma_max`, confirmed independently
by exact Gram elimination. Float64 score abs/rel tolerance1e-9; float32 1e-5 for
small well-conditioned fixtures. Gradcheck eps1e-6, atol1e-5, rtol1e-3.
See reports/m2a/ENGINEER_HANDOFF.md for actual errors, source identity and failures.

## M2-B c-optimality (Issue #18)

_Formerly `docs/M2B_API.md`._

`c_optimality(M, c, *, max_elements=1_000_000)` returns the maximize-oriented
score `-c.T @ M^-1 @ c`. It factors using the unchanged M2-A `_cholesky`
symmetry/tau/pivot policy, solves `L y = c`, and returns `-sum(y**2)` through
the existing finite-output guard. No inverse, ridge, pseudoinverse or clipping.

| Rule | Contract |
|---|---|
| Shapes | M(p,p), c(p,) → scalar; M(B,p,p), c(B,p) → (B,). p,B>=1. No broadcasting, multiple contrasts or L-optimality. |
| dtype/device | Both tensors, float32/float64, identical dtype/device; output preserves dtype/device and autograd. Wrong dtype raises TypeError; shape/device errors raise ValueError. |
| Contrast | Finite and not exactly all-zero per row; zero rejects with first failing batch index. Tiny nonzero contrasts remain valid, although a squared score may underflow in finite precision. |
| Matrix | Same numerical SPD requirement as D/A; singular, indefinite, nonfinite, nonsymmetric or pivot-guard rejected matrices raise ValueError. Near-SPD rejection is numerical policy, not exact singularity proof. |
| Budget | Shapes of M and c checked before factorization. Not a bound on backend workspace or measured peak memory. |
| Finite result | Overflow/nonfinite output rejects. Batched data errors identify the first failing batch index. |

The criterion is differentiable within its accepted SPD domain. With z=M^-1 c,
grad_M=zz.T (on symmetric inputs) and grad_c=-2z. Independent float64 gradcheck
perturbs a free Q in M=Q.T Q+I and nonzero c, keeping matrices in-domain.
CPU oracle tolerance: float64 abs/rel1e-9; float32 abs/rel1e-5 on small
well-conditioned fixtures. Device acceptance and experiments are recorded in
reports/m2b; no universal CUDA/precision/performance claim follows from the API.

### Scientific limitation

Some c-optimal designs are singular (Elfving). For a line model c=(1,1),
the SPD two-point designs {-1,1} and {0,1} both score -1. Replicate design
{1,1} has generalized variance 1/2, but singular information and is rejected
by this v1 criterion. Estimability/range-space checks, generalized inverses,
replicate/approximate-design search and M3 c-optimal search are out of scope.

No existing M1/M2/M3 contract or API is changed. Issue #11 is complete,
Issue #13 remains stopped, and Issue #20 remains queued.

## M3-A Fedorov exchange: CPU numerical search

_Formerly `docs/M3_API.md`._

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

### Numerical search and best-move rule

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
on a directly SPD trial is unresolved.
An unresolved neighbor blocks only if Γ is NaN, Γ>1, or its available direct gain>0; otherwise count `uncertain_nonimproving` and continue.

Tiny positive Δ<=g is recorded separately as marginal, with heuristic upper
gain Δ+g. If a verified best move's lower gain Δ_best-g_best exceeds every
marginal upper gain, accept the best eligible/tied move; otherwise stop
NUMERICAL_INCONCLUSIVE. This avoids an irrelevant roundoff tie blocking a clear
improvement while retaining explicit uncertainty at stopping. The inequalities
are numerical policy only; they do not prove universal best-exchange bounds.

### Results, stopping and resources

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

### Independent evidence

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

## M3-B1 simultaneous two-swap scoring (Issue #11)

_Formerly `docs/M3B1_API.md`._

EXPLORE API/schema frozen before VALIDATE:
`score_two_swaps(F, ids, *, tile_size=64, max_elements=1_000_000)`.
This scores **all legal unordered outgoing position pairs and incoming candidate
ID pairs**, without applying swaps, choosing a winner or running a search.
Only unbatched finite CPU float64 model features F(m,p), CPU int64 unique IDs(n),
m>=n>=p>=1 and an initial Gram accepted by M2-A numerical SPD policy. There is
no k/replication/initializer/search parameter. No RNG consumption or mutation of
inputs. No discrete-selection/autograd guarantee; scoring runs under no_grad.

### Stable schema and enumeration

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

### Mathematics, verification and finite precision

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

### Bounds and scoped oracle evidence

Compute C with Python integers before neighborhood/output allocation. The
budget counts retained tensor output elements 10C+n, fixed major workspace
2mp+6np+10p²+4m+2n+64, and per tile pair16p+128+np+2p². Reduce width to fit the remaining
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

## M3-B2: deterministic k=2 VND

_Formerly `docs/M3B2_API.md`._

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

## M3-C: seeded D-optimal design restarts

_Formerly `docs/M3C_API.md`._

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

### Generic criteria
`optimal_design(F, n, *, criterion="D", restarts=10, generator=None,
max_iter=100, max_elements=1_000_000)` accepts CPU float64 features.
D delegates unchanged to `d_optimal_design`, retaining its fields and adding
`criterion` and `efficiency_lower_bound = d_efficiency_lower_bound`.
A maximizes `-trace(M⁻¹)`; it depends on the supplied feature scaling/coding.
I maximizes `-trace(B M⁻¹)`, with uniform candidate moment `B=FᵀF/m`.
A/I use seeded feasible restarts and directly verified best-improvement 1-/2-swap VND.
Their certified floor is `trace(B M⁻¹)/(n max_j f_jᵀM⁻¹BM⁻¹f_j)` (A: B=I).
They return ids, objective, criterion, efficiency_lower_bound, stop_reason,
certificate, restart_objectives, best_restart and starts.
`LOCAL_2_NUMERICAL` certifies only numerical 1-/2-swap locality, never global
optimality; MAX_ITER/NUMERICAL_INCONCLUSIVE return NOT_CERTIFIED.
### Augmentation
Both design functions accept `fixed_ids=None`, or distinct candidate IDs with length < n.
Each restart draws the remaining IDs uniformly without replacement from non-fixed candidates.
Returned `ids` always contain the fixed IDs; only free positions can leave in an exchange.
Certificates describe locality within this restricted free-position neighbourhood.
The unchanged D/A/I floors bound efficiency against the unrestricted approximate optimum,
so they also conservatively bound efficiency against the augmented exact optimum.
`fixed_ids=None` retains the original seeded starts and exchange paths.
