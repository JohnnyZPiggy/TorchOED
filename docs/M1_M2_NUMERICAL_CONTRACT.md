# TorchOED — M1/M2 Mathematical and Tensor API Contract

> SAE V3.3 · Chief Architect + Scientific Mentor · 2026-10-29 · ARCHITECTURE_ONLY / NOT ENGINEER_DISPATCHED
> Sources: [DEVELOPMENT_PLAN](DEVELOPMENT_PLAN.md), [SCIENTIFIC_MENTOR_RESEARCH](SCIENTIFIC_MENTOR_RESEARCH.md) §§17–25 and [ARCHITECT_ECOSYSTEM_SURVEY](ARCHITECT_ECOSYSTEM_SURVEY.md). E0 actual-wheel evidence in [EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md) §§6–7 and [Draft PR #3](https://github.com/JohnnyZPiggy/TorchOED/pull/3).
> This contract does not alter M0 Issue #1, modify production source, create a new Engineer Issue/PR, run a GPU or authorize merging E0.

## 1. Reuse-First decision / scope

**Core: torch only.** Reuse `torch.linalg`, `torch.quasirandom.SobolEngine`, `torch.meshgrid`/indexing and native autograd. External PyDOE 1.6.0 is a **frozen development-only oracle**, not a runtime dependency; SciPy QMC, OApackage, PyOptEx, OpenTURNS and Pyomo may serve independent *dev-only* oracles. No custom CUDA/Triton, BoTorch/GPyTorch or Pyro dependency is authorized for M1/M2. The 91 exported PyDOE names are a research inventory, not a feature-completeness target.

**Sequencing:** M0 importable torch-only spine → M1 basic candidates/linear & quadratic feature maps → M2 unweighted/weighted SPD information and D/A/E scores → later C/G/I/V once measure/contrast semantics are frozen → M3 exact-ID exchange → later CPU/GPU break-even. Nothing in this document creates Engineer work without a separate Scientist/Architect dispatch.

## 2. Model identities and shapes

| Object | Single case | Batched case | Semantic constraint |
|---|---|---|---|
| Physical candidate factors `C` | `(m,k)` | `(B,m,k)` | Real physical/categorical coordinate representation; **not** model basis |
| Feature map `F=phi(C)` | `(m,p)` | `(B,m,p)` | Explicit intercept and term ordering; quadratic feature count `1+k+k+k(k-1)/2` |
| Exact selected candidate IDs `ids` | `(n,)`, int64 | `(B,n)` | Distinguish **candidate ID** from coordinate equality; `allow_repeats` explicit |
| Selected features `X` | `(n,p)` | `(B,n,p)` | `gather` by ID, not `isclose`-based fuzzy selection |
| Approximate design weights `w` | `(m,)` | `(B,m)` | Each weight nonnegative; sum=1 within declared tolerance; weight is NOT a count until explicitly mapped |
| Information `M` | `(p,p)` | `(B,p,p)` | Maintain exact `normalized` flag, observation/noise convention, rank/regularization diagnostic |
| Criterion output | scalar | `(B,)` | Strict maximize/minimize orientation, validity and reference measure |
| Candidate one-swap gains (later) | `(n,m)` | `(B,n,m)` or tiles | Invalid swaps masked by **index**, not approximate coordinate comparisons |

**Exact independent homoscedastic model:** with `X=F[ids]` and n selected observation events, `M_raw=X.T@X`, `M_normalized=M_raw/n` (n>0). A candidate ID can repeat **only** when `allow_repeats=True`; repeated observational *events* are independent only if the noise model says so. Weighted exact row counts must not be silently normalized to approximate-design weights.

**Approximate design:** with probability weights `w>=0` and `sum(w)=1`, `M(w)=F.T @ (w[:,None]*F)` (batched analog via `einsum`/`bmm`). No explicit dense `diag(w)` in production; no implicit equivalence to n observations without specifying total sample budget n and noise.

**Nonlinear local model** (later): a Jacobian `J(theta,x)` is a derivative of the response with respect to *model parameters*, not the Jacobian of physical factors. Parameter value/shape and observation noise must be explicit. Never confuse the Fisher average of Jacobians with average logdet of per-scenario matrices.

## 3. Covariance and Fisher information: mathematical gate before implementation

For a selected list of distinct or repeated *observation events* S, Jacobian rows `J_S` and a positive-definite observation covariance `Sigma_SS`, define `M=J_S.T @ solve(Sigma_SS,J_S)`. If the known covariance is per-original-candidate, extract the **selected covariance first** and then factorize/solve: `inverse(Sigma_SS) != inverse(Sigma)[S,S]` generally. Scalar noise parameter means **variance** if that contract is selected; diagonal vector entries are variances, not standard deviations. Avoid applying mean-over-scenarios more than once.

**Replication semantic warning:** selecting the same original candidate index twice from a dense covariance can duplicate a row/column and make `Sigma_SS` singular; independent repeat *events* require an explicitly constructed event covariance and are not the same as index duplication. M1/M2 should implement unweighted/diagonal cases first; dense covariance is **DESIGN_ONLY / DEFER** until independent oracle and event identity are frozen. Invalid/indefinite Sigma should be rejected or explicitly diagnosed. No silent ridge.

## 4. Criterion contracts (orientation and required contextual inputs)

| Criterion | Maximized score (default candidate convention) | Required inputs and domain | Numerical primitive |
|---|---|---|---|
| D | `logdet(M)` | M positive definite; rank deficient -> invalid or `-inf` with documented policy | `cholesky_ex`, `slogdet`/triangular solve |
| A | `-tr(M^{-1})` | M positive definite; no hidden parameter scaling | Cholesky solves against identity, trace |
| E | `lambda_min(M)` | Symmetric M; repeated min eigenvalue means nonunique eigenspace/subgradient | `eigvalsh` / rank diagnostic |
| C | `-c.T@solve(M,c)` | **contrast vector c mandatory**, not implicit ones | Cholesky solve |
| I | `-tr(solve(M,H))` | **moment matrix H mandatory** with defined prediction-design measure; H=`E[ff.T]` | Cholesky solve / weighted trace |
| V | `-mean_j (g_j.T@solve(M,g_j))` | Explicit finite **evaluation feature rows g_j**, optional weights | batched triangular solves and reduction |
| G | `-max_j (g_j.T@solve(M,g_j))` | Explicit prediction candidate set / domain; never confuse a grid approximation with global domain supremum | tiled solve/max |
| S | **DEFER** until independently fixed | Upstream score is NOT automatically det(correlation) | research oracle first |
| T | **DEFER** until two-model discrimination objective fixed | Projection vs residual difference; cannot use zero-model default as useful science | independent model-discrimination oracle |

**Important statistical distinction:** parameter covariance convention includes a possible noise-variance multiplier; the score above assumes M already incorporates precision as agreed. D-opt raw determinant, normalized determinant and D-efficiency percentages are *different metrics*, and across different n/p can rank designs differently. No output should be called a percentage efficiency without an explicit reference and formula. All listed signs are defined for maximization; an API may expose an `orientation` field for interpretability.

## 5. Validation & numerical accuracy contract

- **Shapes/dtypes:** base oracle float64 CPU; torch.float32 separately with reported errors; integer IDs int64. Batch B dimension is explicit, never silently broadcast different candidates. Device preserving operations; CUDA semantics are architecture hypothesis, NOT_RUN.
- **No implicit regularization:** singular/indefinite M must yield a predictable status and/or exception; any requested ridge `lambda>0` creates a separately named *regularized* objective. Check `cholesky_ex` `info` before using a factor. Unknown behavior (NaN/Inf) is invalid, never a silent PASS.
- **Diagnostics:** record symmetry residual, rank/condition estimate where practical, Cholesky success, normalization, n, p, dtype and device. Do not mandate expensive SVD for every production call: the independent oracle can do so for small problems.
- **Score tolerance:** for well-conditioned float64 cases use `atol=1e-9, rtol=1e-9` initially, then tighten for exact small fixtures. For ill-conditioned inputs stop/flag INCONCLUSIVE instead of using an arbitrary tolerance to force PASS. Swap-improvement threshold must account for objective scaling; fixed `1e-12` absolute threshold is inadequate (Mentor §22).
- **Autograd:** derivative for SPD simple eigenvalues and Cholesky solutions can be tested using `torch.autograd.gradcheck` on float64; at repeated eigenvalues / singular boundaries gradient can be undefined/nonunique. Never claim gradient parity there.
- **Tie/selection semantics:** stable first-index or lexicographic tie rule specified at each API; sample count and RNG `torch.Generator` state explicit; no exact SciPy-vs-torch scrambled sequence promise.
- **Security/portability:** no NumPy/SciPy/PyDOE in production numerical hot path; dev comparator isolated, dependency licenses/NOTICE retained. No Actions or new paid resource commitments.

## 6. Independent CPU Oracle matrix and falsifiable M1/M2 acceptance

| ID | Exact small fixture | Expected independent mathematical result | Gate |
|---|---|---|---|
| A1 | 1 factor, `C=[[-1],[0],[1]]`; degree-1 features | `F=[(1,-1),(1,0),(1,1)]`, `(3,2)`; selected IDs `[0,2]` -> raw M=diag(2,2), normalized M=I | exact element match |
| A2 | 2 factors, full 2-level grid | 4 points, 6 quadratic columns: `[1,x,y,x²,y²,xy]`; degenerate binary-square rank reported | ordering/shape/rank explicit |
| A3 | `C` contains two distinct IDs with identical coordinates | `ids=[0,1]` legal in without-replacement mode; `ids=[0,0]` illegal there, allowed only with replicates | exact index logic |
| A4 | Same feature basis `F`, approximate w=(1/2,0,1/2) | weighted M=I, equality to exact normalized M of `[0,2]` only under assumed homogeneous independent observations | analytic equality + semantic warning |
| A5 | `M=diag(2,8)`, `c=[1,2]`, `H=diag(3,5)`, `g=[1,1]` | D=log16; A=-(1/2+1/8)=-5/8; E=2; C=-(1/2+4/8)=-1; I=-(3/2+5/8)=-17/8; g prediction variance=1/2+1/8=5/8 | exact scalar |
| A6 | `M=diag(1,0)` | D/A invalid or policy-specific `-inf`/exception, never finite SPD success; rank 1 | negative-case status |
| A7 | Full covariance with off-diagonal correlation and selected indices subset | `solve(Sigma[S,S],J[S])` agrees with independent direct NumPy/SymPy; `inverse(Sigma)[S,S]` demonstrably differs | exact selection-before-inversion |
| A8 | Two scenario FIMs `diag(100,1)` and `diag(1,100)` | `logdet(mean FIM)=log(50.5²)` vs `mean logdet FIM=log(100)` unequal | no illegal averaging interchange |
| A9 | P1 E0 actual small-composite 20×4 coordinates | degree-2 features `(20,15)` **rank 15**, exact Gram determinant `18014398509481984`; separate `D=ABC` cube rank 12, Gram determinant 0 | source-vs-math positive/negative control |
| A10 | batched `B=2`, unbatched independent formula on each | batched information/scoring numerically matches two scalar oracles with declared broadcasting | float64 1e-9 initially |

**E0 lesson:** original PyDOE 1.6.0 `small_composite_design(4,center=(2,2))` has **full rank 15**, not the earlier hypothesized 12, because the actual fractional cube uses `D=AB`, with axial points breaking some aliases. See Mentor §25; this is an actual executed-wheel result reported in PR #3 and independently reconstructed from its stored coordinates, not a claim of full upstream parity.

## 7. CPU / GPU break-even experimental *plan*, not results

Three tiers: `(B,m,n,p)=(1,32,8,4)`, `(8,512,32,16)`, `(8,4096,64,32)`. Compare *same* candidate IDs/feature matrices and score/stop semantics across independent reference NumPy CPU, torch CPU direct recompute, torch CPU reused factorization/batched candidate tiles, torch CUDA only **after** GPU device/WARM capability and separate approval. Measure real end-to-end wall time incl transfers/launch/sync, peak memory, correctness, rank failures and allocation. A speedup on standalone GEMM is not evidence for the full OED workload. No performance assertion before hardware results.

## 8. ADR and authority / Engineer handoff

| Decision | State |
|---|---|
| Torch-only runtime / reuse linalg and Sobol | ACCEPT ARCHITECTURE |
| M1 candidates + polynomial features before M2 optimization | ACCEPT SEQUENCE |
| M2 D/A/E and independent small CPU oracles first | ACCEPT SEQUENCE |
| C/G/I/V require explicit contrast, moment and evaluation-domain inputs | ACCEPT MATHEMATICAL CONTRACT |
| Dense correlated covariance and replicated event semantics | DEFER IMPLEMENTATION |
| S/T score, k-exchange, Smolyak quadrature | DEFER until separately defined & independently verified |
| GPU acceleration and cuOED | DEFER until true device evidence and break-even |
| M0 Issue #1 scope | UNCHANGED; no Engineer dispatch by this document |
| E0 Draft PR #3 | EXPERIMENT_ACCEPTED; pending separate Scientist merge approval |

**Next Architect gate:** when an M0 Engineer PR appears, independently review import/install/torch-only dependency, 91-name inventory, minimal smoke and explicit limits. Then Scientist may authorize discrete M1 work. This contract provides the mathematical definitions and independent A1–A10 targets, but **must not be interpreted as a request to modify M0**. No new issue, PR or merge is created here.
