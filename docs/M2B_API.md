# M2-B c-optimality (Issue #18)

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

## Scientific limitation

Some c-optimal designs are singular (Elfving). For a line model c=(1,1),
the SPD two-point designs {-1,1} and {0,1} both score -1. Replicate design
{1,1} has generalized variance 1/2, but singular information and is rejected
by this v1 criterion. Estimability/range-space checks, generalized inverses,
replicate/approximate-design search and M3 c-optimal search are out of scope.

No existing M1/M2/M3 contract or API is changed. Issue #11 is complete,
Issue #13 remains stopped, and Issue #20 remains queued.
