# M3-B Architecture Gate — Modified Fedorov, DETMAX and Genuine k-Swap

**SAE V3.3 · Chief Architect + Scientific Mentor · 2026-10-29**

**Status: MATHEMATICAL_DESIGN_ONLY / CPU_ORACLES_SPECIFIED_NOT_EXECUTED / ENGINEER_NOT_DISPATCHED.** Baseline `main=573ab0d60bcf1a5cd61bbdb2c43a08b07e47444f`, with M3-A exact-ID Fedorov CPU merged via PR #10. E0 Draft PR #3 remains independent. This document creates no Engineer Issue, CLAIM, PR, production implementation, Actions or GPU workload.

Sources of existing mathematics: [Scientific Mentor Research §§11–14](SCIENTIFIC_MENTOR_RESEARCH.md) (especially four-point trap and 2k×2k determinant lemma), [M3-A gate](M3_FEDOROV_EXCHANGE_GATE.md), [M3 API](M3_API.md), [M1/M2 numerical contract](M1_M2_NUMERICAL_CONTRACT.md), and frozen PyDOE source comparisons recorded by Mentor. `sae.yaml` pins SAE V3.3. **Do not conflate algorithms sharing the names Modified Fedorov or DETMAX across implementations.**

## 1. Model, identities and target

Use finite `F[m,p]`, exact candidate IDs `S⊂{0,…,m-1}`, no replacement, fixed size `n`, `m≥n≥p`, `M(S)=Σ_{j∈S}f_j f_jᵀ ≻0`, and maximize `D(S)=logdet M(S)`. Candidate IDs, coordinate equality, row/event position, and replicate multiplicity are distinct. M3-B mathematical reference is unbatched float64 CPU with *independent exact arithmetic* tiny oracles; GPU, with-replacement, dense covariance, nonlinear Fisher and autograd-through-discrete-selection are deferred.

Preserve M3-A's observed numerical policy: SPD and condition checks with *no implicit ridge*. Any finite-precision local result must be labeled numerical unless all neighbor inequalities have certified error enclosures.

## 2. Four non-equivalent search mechanisms

| Family | Precise update/neighborhood | What can escape strict 1-swap traps? |
|---|---|---|
| **Fedorov M3-A** | Best strictly improving one-for-one exchange, recompute neighborhood after acceptance | No, from same starting S |
| **Modified Fedorov** | Sweep outgoing positions; each can make an improving *single* exchange against current design; update state immediately | **No**, if every 1-swap neighbor is strictly worse, regardless of sweep order |
| **Single-level add/drop** (observed frozen PyDOE detmax variant) | Select one add candidate, then drop one of the enlarged design to return to n | **No**, any resulting fixed-n set is S or a 1-swap neighbor |
| **Genuine simultaneous k-swap / multi-level DETMAX excursion** | Replace k≥2 selected IDs together, or traverse n→n+k→n (and optionally n−k), allowing non-monotone/intermediate-size states | **Can**, but only with suitable search policy; a reachable path does not guarantee a particular greedy rule finds it |

**Semantics caution:** a Modified Fedorov sweep may accept several *sequential* 1-swaps, but it is not a simultaneous k-swap and cannot certify k-locality. Classical multi-level DETMAX and PyDOE's frozen one-level add/drop must have **different API names or explicit excursion depth**. No claim of identical source parity.

## 3. Exact k-swap mathematics and complexity

Choose outgoing index set `O⊂S`, incoming `N⊂Sᶜ` of equal cardinality k. Let selected feature-row matrices be `A_O[k,p]`, `B_N[k,p]`, and `W=[A_Oᵀ,B_Nᵀ] ∈ R^{p×2k}`, `J=diag(-I_k,+I_k)`. Then:

`M'=M+W J Wᵀ` and `Γ_k=det(I_{2k}+J Wᵀ M^{-1}W)=det(M')/det(M)`.

The equivalent blocks are `[[I-A_O M^{-1}A_Oᵀ, -A_O M^{-1}B_Nᵀ],[B_N M^{-1}A_Oᵀ, I+B_N M^{-1}B_Nᵀ]]`. The sign/order of J matters. Evaluate solves using Cholesky of **original** M; deletion-only intermediate can be singular when n=p. Determinant ratio positive alone is insufficient to establish positive-definite M'; verify prospective complete M' with an independent SPD/QR reference before accepting. No clipping Γ, hidden ridge, or comparing variable-size raw determinants as though n were fixed.

For fixed exact k the full neighborhood size is `C(n,k)C(m−n,k)`; full union of 1…K neighbors adds these counts, with `k≤min(n,m−n)`. For k=2: Θ(n²(m−n)²) swap combinations, before score and verification costs. A direct tiny independent oracle builds each prospective `X'ᵀX'` (O(np²+p³) per case); low-rank `2k×2k` scoring costs additional small matrix algebra, plus shared factorization/triangular solves. Report actual exact accounting for duplicates, memory tiles and full-end-to-end operations before any performance claim.

Any distinct fixed-size design T is obtainable from S with k=|S\T| swaps; searching up to k=min(n,m−n) covers all n-subsets (combinatorial explosion). Single-level add/drop changes at most one original membership ID and **cannot** leap a strict 1-swap trap.

## 4. Fixed four-point local-trap oracle

`F = [(1,0),(0,1),(9/10,9/10),(9/10,-9/10)]`, `n=p=2`, start `S={0,1}`.

Every 1-swap neighbor has `det M=81/100 < 1=det M(S)`. The genuine 2-swap `S'={2,3}` gives `M'=diag(81/50,81/50)`, `det M'=6561/2500=2.6244`. Hence `ρ₁(S)=1` and `ρ₂(S)=6561/2500`. This proves one-swap local ≠ two-swap/global. A sequential Modified Fedorov sweep and single-level add/drop, both enforcing strict fixed-n improvement, must stay at S; exhaustive 2-swap can jump to S'.

A two-add/two-drop excursion `{0,1}→{0,1,2}→{0,1,2,3}→{1,2,3}→{2,3}` reaches the improved design, **but** changing n changes raw determinant semantics. Its mere existence is *not* evidence that a named DETMAX heuristic chooses the path. Independent exact subset enumeration over all six 2-sets is the global oracle.

## 5. Proposed independent CPU oracles M3B-B1–B12 (NOT_RUN)

| Gate | Input / method | Falsifiable expected property |
|---|---|---|
| B1 | Four-point rational trap, full 1-neighborhood | Each four neighbor det=81/100; initial 1-swap local |
| B2 | Same trap, exact simultaneous k=2 | Gain ratio=6561/2500; global optimum by six exact 2-subsets |
| B3 | Same trap, sequential Modified Fedorov with strict positive gains only | No move from {0,1}; sweep ≠ k=2 |
| B4 | Same trap, any single-level add-one/drop-one final set | At most one original ID replaced; cannot reach {2,3} |
| B5 | Explicit 2-add/2-drop path | Distinct-size intermediate steps valid; endpoint ratio correct; no claim specific greedy DETMAX finds path |
| B6 | k=1 random small full-rank integer F | 2k determinant lemma matches direct exact Gram ratios for all legal swaps |
| B7 | k=2 small integer F, n=6, m=11 | 15×10=150 legal simultaneous pairs; direct rational det ratio equals rank-4 lemma within stated float64 tolerance; count corrected for n=6,m−n=5 |
| B8 | n=p saturated case, k≥1 | Original M factorization is valid; deletion-only can be singular; full k-swap checked directly |
| B9 | Distinct IDs sharing identical feature rows | IDs remain distinct, deterministic ties/neighbor counts not deduplicated by coordinates |
| B10 | Deterministic tie + seed/restarts | Exact lexicographic outgoing-ID tuple then incoming-ID tuple tie; distinct starts recorded; seeded replays, no claim restart expands per-state neighborhood |
| B11 | Monotonicity and termination, bounded sweeps/max_iter | Strict verified improvements for fixed-size accepted endpoints; explicit MAX_ITER / NUMERICAL_INCONCLUSIVE / LOCAL_k_NUMERICAL / NO_LEGAL_SWAP; no unearned exact certificate |
| B12 | Singular/near-singular new M and pathological Γ | Reject true singular, preserve uncertainty for near-singular SPD, check no ridge/NaN clipping; ratio sign not SPD proof |

**Important count correction:** previously cited `825` pairs for `p=3,n=6,new=11,k=2` refers to **11 incoming candidates in addition to 6 selected** (`m=17`, count `C(6,2)C(11,2)=825`). If `m=11` **total**, incoming=5 and only `150` pairs. The acceptance fixture must label whether a number describes total rows or incoming rows. B7 uses m=11 total and expects 150; an optional separate m=17/incoming=11 fixture expects 825.

For small integer/rational features use Python `fractions.Fraction`, exact integer Bareiss determinant and `itertools.combinations` to enumerate all `C(m,n)` designs, derive a full `N_k(S)` and compare literal determinants. This reference must not import TorchOED's optimized swap-score implementation or reuse its computed expected values. Cross-check k=2 rank-4 lemma against direct rational ratios on every tiny admissible neighbor. The search algorithm may return 1-local but not global; record as expected **negative control**, not a failure.

## 6. Stopping, validity and numerical certificates

- **Fixed-n Modified Fedorov:** one pass visits each outgoing *position* against the current eligible IDs; acceptance changes current state immediately; deterministic stable ordering, pass count and an optional full one-swap closing scan. `NO_IMPROVEMENT_IN_SWEEP` without a full complete best-neighbor scan is not a universal 1-local certificate if the implementation skips candidates.
- **Simultaneous k-swap:** require a complete declared `N_{≤K}` scan before claiming numerical K-locality. `LOCAL_K_NUMERICAL` means exhaustive within floating tolerance/policy, **not exact mathematical optimality**. `EXACT_CERTIFIED` only with independently proven error intervals or exact rational comparisons for a truly small all-rational fixture.
- **Multi-level excursion:** specify allowed cardinality range `[n−d,n+d]`, legal intermediate SPD/shape constraints, maximum excursion depth, deterministic add/drop policy, visit-cycle prevention, and final fixed-n objective check; intermediate logdet of differing cardinalities must not be used to certify final improvement. Stop reasons include `NO_ENDPOINT_IMPROVEMENT`, `MAX_EXCURSIONS`, `NUMERICAL_INCONCLUSIVE`.
- **Global optimality claim:** only from enumeration of *all* `C(m,n)` designs (tiny exact arithmetic) or separately derived valid global upper bound. Restarts and fixed top-K pruning alone cannot certify global or full K-local optimality.
- **Efficiency bound (separate optional scientific diagnostic):** normalized `Mbar=M/n`, `d_max=max_j f_jᵀ Mbar^{-1}f_j`; for any exact n-row comparator, `logdet M*−logdet M ≤ p log(d_max/p)`. Report assumptions and distinguish this upper bound from actual achieved optimality. For four-point trap, bound ratio 2.6244 is tight.

## 7. Architecture decision and work boundaries

**Proposed later engineering order, not authorized now:** (i) M3-B1 independent k=2 *scoring and exact tiny oracle*, no search overhaul; (ii) bounded deterministic k=2 search; (iii) Modified Fedorov sweep semantics; (iv) explicit-depth DETMAX excursions. Do not package all methods under an ambiguous single `detmax` flag. Reuse M3-A exact-ID, direct D check, limits, records and structured stopping. Production may use `torch.linalg`, but PyDOE/SciPy/NumPy only as isolated dev comparators with source provenance; never call a CPU float64 test GPU evidence.

**Current gate:** `M3B_MATH_AND_ORACLES_FROZEN / M3B_ENGINEER_NOT_DISPATCHED / ALL_M3B_RUNTIME_TESTS_NOT_RUN`. Any Engineer Issue/CLAIM, source implementation, GPU test or merge requires separate Scientist authorization.
