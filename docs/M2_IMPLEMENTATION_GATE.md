# M2 Information Matrices & Criteria — Architect implementation gate

**SAE V3.3 / Scientific Mentor, 2026-10-29.** `ARCHITECTURE_ONLY / M2_NOT_DISPATCHED`. This is a bounded future task specification, **not** a Codex CLAIM or Engineer authorization. Scientist must separately authorize M2 implementation/CPU execution. Upstream: [M1/M2 numerical contract](M1_M2_NUMERICAL_CONTRACT.md) §§2–6 and [release roadmap](RELEASE_ROADMAP.md). Baseline main at drafting: `cc9e9991b44e823f7f55e924e602527799ccc07f` (Scientist-authorized M1 merged). E0 PR #3 remains unmerged; its frozen P1 coordinates are dev-only oracle.

## 1. Strict first slice (M2-A) and non-goals

Deliver *only*:
1. `exact_information(F, ids, *, normalized=False, allow_repeats=False)`: 2D `F[m,p]`, 1D `ids[n]` with explicit 3D/2D batched counterpart; return `X.T @ X` where `X=F[ids]`. If normalized, divide by `n`, with `n>0`; do **not** confuse indices with coordinate uniqueness. IDs must be int64, in range, no duplicate unless explicitly allowed. For `n=0`, raw information may be zero `p×p` with explicit policy; normalized mode must reject.
2. `weighted_information(F,w)`: `F.T @ (w[:,None]*F)` (batched analog), `w>=0` and weights sum to 1 within frozen dtype-scaled tolerance. Reject negative, NaN/Inf, wrong shapes and unnormalized weights. No dense `diag(w)`; no silent conversion from weights to counts.
3. `d_optimality(M)`, `a_optimality(M)`, `e_optimality(M)` with **maximization convention**: D=`logdet M`; A=`-tr(M^{-1})`; E=`lambda_min(M)`. Strictly symmetric, finite input matrix; positive-definiteness required for D/A. A uses Cholesky + solves, never explicit inverse or silent ridge. E on PSD singular matrices returns 0; for indefinite matrices reject by policy (not silently report a valid information score). Maintain scalar/(B,) output, dtype/device preservation; no benchmark claims.

**Deferred**: nonlinear Jacobian/Fisher with correlated/dense covariance; implicit regularization; C/I/V/G/S/T criteria; optimization/exchange search; general fractional/physical observation noise models; GPU speedups, Actions and PyDOE numerical backend.

The generic word “Fisher” in marketing is *not* an implementation claim of noise-aware nonlinear Fisher until separate noise/Jacobian contracts are implemented. Homoscedastic linear information is the first slice.

## 2. Mathematical invariants & failure contracts

- Normalize only on explicit request; raw `M` scales linearly in sample count, normalized `M` does not. Repeated same ID under `allow_repeats=True` means separate independent observation events **only within the declared homoscedastic model**.
- Check shapes upfront, including `p>=1`, finite `F`, floating dtype float64 CPU initially and separately float32; disallow implicit broadcasting of `B` across independent cases. Reject bool IDs, non-int64 IDs and out-of-range values; use exact IDs, never coordinate `isclose`.
- For `M`, inspect symmetry residual with dtype/scale-aware tolerance and reject non-symmetric; check `torch.linalg.cholesky_ex` info for D/A. D=2 sum(log diag(L)); A negative trace of `cholesky_solve(I,L)`. A/D reject singular/indefinite; no regularization. E=`eigvalsh(M)[0]` on symmetric PSD and warn/document repeated eigenvalue non-unique gradients. Reject invalid/nonfinite.
- API decision to freeze before Codex implementation: choose **exceptions (ValueError for invalid shape, weights, singular SPD)** rather than `-inf` or silent NaN for first slice; if batched, report invalid batch index in error. No accepting a mixture of finite PASS and invalid matrices silently.
- Dense covariance model is not equivalent to submatrix of inverse: `(Sigma_SS)^{-1}\neq(Sigma^{-1})_{SS}`. Future distinct observation-event IDs must be explicit.

## 3. Independent CPU oracle gates: proposed M2-A1–A12

| ID | Independent mathematical fixture | Exact/invariant expected result |
|---|---|---|
| A1 | `F=[[1,-1],[1,0],[1,1]]`; `ids=[0,2]` | raw `diag(2,2)`; normalized `I_2` |
| A2 | `ids=[0,0]` and repeated coordinate at distinct IDs | duplicate ID rejected unless `allow_repeats=True`; distinct IDs legal irrespective of equal coordinates |
| A3 | same `F`, `w=(1/2,0,1/2)` | weighted `M=I_2`, but not identical observation semantics to raw exact sample counts |
| A4 | `M=diag(2,8)` | D=`log(16)`, A=`-5/8`, E=`2` |
| A5 | `M=diag(1,0)` | D/A fail explicitly, E=`0` on PSD |
| A6 | `M=diag(1,-1)` | D/A/E reject invalid indefinite information |
| A7 | `M=[[2,1],[1,2]]` | D=`log 3`; A=`-4/3`; E=`1` |
| A8 | `X` repeated with 2x independent events | raw information doubles; normalized information invariant |
| A9 | weights with negative/NaN/Inf/nonunit sum, and `ids` OOB/nonint64 | all rejected before numerical computation |
| A10 | `B=2` batched independent F/ids/w/M | output equals stacked scalar oracle with no unintended cross-batch broadcast |
| A11 | float64 `gradcheck` for SPD D/A and *simple* smallest-eigenvalue E | gradients agree with analytical finite-difference; repeated eigenvalue E gradient NOT_RUN/undefined |
| A12 | E0 P1 frozen 20×4 physical C mapped by M1 quadratic features to 20×15 F | raw full-design Gram exact determinant `18014398509481984` and rank 15; separate D=ABC fixture rank12 and D/A invalid |

Tolerances: exact IDs and integer-Gram fixtures require exact equality; well-conditioned float64 `atol=rtol=1e-9` initially; float32 tolerance separately justified; numerical rank thresholds declared rather than retrofitted. For `A12`, determinant may be checked with independent exact rational/integer method and `logdet` against `log(18014398509481984)`; **do not** treat a positive determinant as a substitute for Cholesky success.

## 4. Independent tests vs self-referential tests

Test expected 2×2 matrices and scalar results from literal mathematics (or exact Fraction/SymPy in dev-only oracle); never call the same production implementation twice to generate its own expected value. Use numerical properties to supplement, not replace, explicit fixtures: symmetry, PSD for information, positive scaling, batch parity, correct rank and finite rejection. Keep true E0 source identity: `tests/fixtures/e0_p1.json` from PR #6 records actual real-wheel coordinates and its frozen E0 provenance. Do not substitute D=ABC for that source.

## 5. Proposed Engineer deliverables once Scientist dispatches

- One Issue (only after searching for existing M2 task) → one COLD/CPU WARM CLAIM → one feature branch and Draft PR, no self-merge.
- Files limited to information/criterion torch modules, their tests, documented exports and small M2 evidence report. Keep M0/M1 source and E0 PR #3 independent; do not expand to M3 search.
- Actual Python/torch/platform, source/tested/final SHAs, complete changed-files SHA-256 manifest, logged bounded CPU test/install/wheel/external-import commands, test counts and failure history, retained independent exact oracles, and `ENGINEER_HANDOFF`.
- Formal Architect acceptance only after independent source/math/evidence inspection; Scientist alone approves future merge.

**Current gate:** `M2_MATH_AND_CPU_ORACLES_SPECIFIED / M2_ENGINEER_NOT_AUTHORIZED`. No code, no new Issue/PR, no GPU or paid service was initiated by this document.
