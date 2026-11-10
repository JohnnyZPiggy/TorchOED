# P1.5 Gate — Model features (M1) and ask/tell (M2)

**Chief Architect · 2026-11-10 · Week 5 goals M1, M2.** Status: `P15_MATH_FROZEN` — independent fresh-context review **PASS** (2026-11-10, wording fixes applied, recorded in the merging PR). Baseline `main=98e5744`.

Notation as in the [W1 gate](W1_GENERIC_CRITERIA_GATE.md): candidate feature rows `F[m,q]`, exact IDs, no replacement, `M = F_SᵀF_S`. Labels: **PROVED** = derived here and checked numerically (Fraction or float64, Architect sandbox and independent reviewer); **OBSERVED** = finite evidence only; **CONJECTURE** = not established.

## 1. Locally linearized design (M1)

Model `y = f(x, θ) + ε`, `θ ∈ ℝ^q`, errors independent, mean 0, common variance `σ²` (homoscedastic). Nominal value `θ₀` supplied by the user. Row `J(x) = ∂f(x,θ)/∂θ |θ₀` (`1×q`).

1. **Information (PROVED, standard).** For Gaussian errors, the Fisher information of an exact design `S` at `θ₀` is `σ⁻² M(θ₀)` (for general least squares it is the inverse asymptotic covariance; for nonlinear `f` only asymptotically), `M(θ₀) = Σ_{i∈S} J(xᵢ)ᵀJ(xᵢ)`. With `F[i] = J(xᵢ)` this is exactly the existing `M = F_SᵀF_S`.
2. **σ² is irrelevant to the argmax (PROVED).** `logdet(σ⁻²M) = logdet M − q log σ²`; `tr(B(σ⁻²M)⁻¹) = σ² tr(BM⁻¹)`. D/A/I optimal IDs do not depend on σ².
3. **Decision: `optimal_design` is used unchanged.** Rows of `jacobian_features` are passed as `F`. All certificates and efficiency floors returned are functions of `F` only, hence they are statements about the **linearized problem at θ₀ over the given candidates**, not about the nonlinear model at the true θ. Docs and docstring must say so.
4. **Linear model (PROVED).** If `f(x,θ) = θᵀφ(x)` then `J(x) = φ(x)` for every `θ₀`. Oracle (a): for `f = polynomial_features(x, d)·θ`, `jacobian_features` equals `polynomial_features(x, d)` to `1e-12` (float64).
5. **Reparametrization (PROVED; Fraction, 1293 nonsingular designs pooled over 40 random integer `F` and invertible `T`, `m=7, q=3, n=4`; reviewer: 2061 over 60).** For `F → FT`: `det M` scales by `det(T)²` (D design unchanged), `tr(W M⁻¹)` with `W = FᵀF/m` recomputed from the transformed `F` is invariant (I design unchanged; a fixed user-supplied `W` would not be), `tr(M⁻¹)` is invariant only for orthogonal `T` (0/1293 equal for random `T`): A designs depend on the parametrization in general. Consequence: if `f` is linear in a parameter `c` that (`c ≠ 0`) only scales a column of `J` (e.g. `a` in `a·e^{−bx}`, `V` in Michaelis–Menten), the local D and I designs do not depend on the nominal value of `c`.
6. **Exponential decay (PROVED).** `f = a·e^{−bx}`, `x ∈ [0,T]`, `a ≠ 0, b > 0`: `J = (e^{−bx}, −a x e^{−bx})`, and for two points `|det| = |a| e^{−b(x₁+x₂)} |x₂−x₁|`. For `x₁<x₂` it decreases in `x₁` ⇒ `x₁ = 0`; `x₂ e^{−bx₂}` is maximal at `x₂ = 1/b`. Hence the 2-point local D design is `{0, 1/b}` when `T ≥ 1/b`. Float64 grid check (2001 points, three `(a,b,T)`): argmax within one grid step of `{0, 1/b}`.
7. **Michaelis–Menten (PROVED).** `f = Vx/(K+x)`, `x ∈ (0, x_max]`, `V,K > 0`: `J = (x/(K+x), −Vx/(K+x)²)`, two-point `|det| = V x₁x₂(x₂−x₁) / ((K+x₁)²(K+x₂)²)`. `∂/∂x₂ log = 1/x₂ + 1/(x₂−x₁) − 2/(K+x₂) > 0` ⇒ `x₂ = x_max`; the unique interior critical point of `log x₁ + log(x_max−x₁) − 2 log(K+x₁)` is `x₁ = K x_max/(2K + x_max)`, and the determinant → 0 at both ends, so it is the maximum. Float64 grid check (4000 points, three `(V,K,x_max)`): argmax within one grid step.
8. **Emax with baseline (OBSERVED).** `f = E₀ + E_max x/(ED₅₀+x)` on `[0, x_max]`: the 3-point local D design found by exhaustion on a 101-point grid is `{0, ED₅₀ x_max/(2ED₅₀+x_max), x_max}` for two parameter sets. Used only as a documentation example, never as a test.
9. **Last layer of a network (PROVED).** If `f(x) = wᵀh(x) + c` with `h` the penultimate activations, then `J` w.r.t. `(w, c)` is `(h(x), 1)`, independent of `w, c`: last-layer design is a classical linear design on learned features. With all parameters, `J` depends on `θ₀` and `q` can exceed `n`; `optimal_design` then raises (`m ≥ n ≥ q` is required) — no regularization is added.
10. **Replication.** IDs are drawn without replacement, so a design with `n > q` spreads over distinct candidates instead of replicating support points. Replicates are obtained by duplicating candidate rows; no new mechanism.

### API decision: `jacobian_features(model, X, *, params=None, max_elements=1_000_000)`

- `model`: a `torch.nn.Module`; `model(X)` must return shape `(m,)` or `(m,1)` (scalar response; validated, it is a trust boundary). `X`: tensor with leading dimension `m`.
- `params=None`: all parameters with `requires_grad`, in `named_parameters()` order. Otherwise a sequence of parameter names (e.g. `["head.weight", "head.bias"]`); columns follow the given order. Each parameter is flattened with `reshape(-1)`. Unknown names raise.
- Nominal `θ₀` = the module's current parameter values. The module is not modified; no optimizer state.
- Return `F[m, q]`, detached, in the model's dtype. Computation: `torch.func.jacrev` over `torch.func.functional_call` w.r.t. the selected parameters only; output memory `m·q` checked against `max_elements` before computing.
- Row `i` is `∂ model(X)[i] / ∂θ`. If the module couples rows (e.g. BatchNorm in training mode) that is still the returned derivative; the docstring tells users to call `model.eval()`.
- Complexity: one reverse-mode pass per output row (vectorized by `jacrev`), `O(m · cost(f))` time, `O(m·q)` output memory.

## 2. Ask/tell (M2)

**API decision: `next_batch(F, run_ids, k, *, criterion="D", restarts=10, generator=None, **optimal_design kwargs)`** returns a sorted `int64` tensor of `k` new candidate IDs.

1. **Definition (PROVED by construction).** `S = optimal_design(F, len(run_ids)+k, criterion=…, fixed_ids=run_ids, …)["ids"]`; return `S \ run_ids`. Since `optimal_design` keeps every fixed ID and draws without replacement, the result has exactly `k` IDs, none in `run_ids`.
2. `run_ids` empty ⇒ plain `optimal_design(F, k, …)`. Preconditions are those of `optimal_design` (`m ≥ len(run_ids)+k ≥ q`, distinct valid IDs); no extra checks.
3. **Determinism.** Same `F`, `run_ids`, `k`, seeded `generator` ⇒ identical output (inherits `optimal_design` determinism).
4. **Certificates.** Objective, efficiency floor and certificate describe the full augmented design `S` conditional on the fixed `run_ids`, for the current linearization; they are not a guarantee about the `k` new points alone nor about an unconstrained optimum.
5. **Closed loop (M3) uses it as:** fit `θ̂` → `F = jacobian_features(model at θ̂, X_candidates)` → `next_batch(F, run_ids, k)`. `F` is recomputed every round; previously run IDs stay in `run_ids`. Certificates are for the current linearization only.

## 3. Held-out additions (frozen 2026-11-10; never used for tests, tuning or dispatch choices; run once at week end, reported as-is)

Rows by `jacobian_features` at `θ₀` on the stated float64 grid (`torch.linspace`), `optimal_design` defaults, `generator=torch.Generator().manual_seed(20261110)`; report IDs, objective, efficiency floor, stop reason, per-restart time.

| Case | Model, `θ₀` | Candidates | n | Criteria |
|---|---|---|---|---|
| **H9** | Emax `E₀ + E_max·x/(ED₅₀+x)`, `(E₀,E_max,ED₅₀) = (1, 2, 0.5)` | `linspace(0, 4, 401)` | 9 | D, I |
| **H10** | `a·e^{−bx} + c`, `(a,b,c) = (1, 1.5, 0.2)` | `linspace(0, 6, 601)` | 8 | D, A |

## 4. Not in this gate

Heteroscedastic or non-Gaussian noise, vector outputs, Bayesian/minimax robustness over θ (P3), continuous (gradient) design over `x`, BoTorch/Ax adapters.
