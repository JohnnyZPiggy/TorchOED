# M2-A CPU information and criteria

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
