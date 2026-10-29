# M1 CPU constructor contract

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
