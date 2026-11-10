# Week 5 goal — 2026-11-10 → 2026-11-17 (VISION.md phase P1.5: model-driven closed loop)

**Opened early by Scientist decision (2026-11-10, in chat with the Architect):** week 4 work is finished, so week 5 starts now instead of waiting for 2026-11-14 and a 24 h week-goal window. Week 4 result: C1 A/I speed **closed, not met, 5/8** (`reports/w168`; per the 2026-11-09 time-box, no carry-over; further A/I speed only via Mentor #125 or ≤ 1 background S task per week), C2 met (#126), C3 met (#129, CI pytest ≤ 12.1 s), C4 blocked on the Scientist (#128). The week-4 held-out record #176 runs now (START at `68ad258`) and is reported as-is. Baseline: `main@68ad258`, v0.6.0.

**Why this week:** the Scientist's direction (VISION.md P1.5) is a torch-native classical OED method that consumes any PyTorch model: Jacobian features feed the existing D/A/I engine, an ask/tell loop reuses `fixed_ids`, and a simulated lab backend shows the closed loop. Positioning: small-budget, constrained, certified experiments, not a competitor to active-learning methods. The scale crossover experiment answers, with data, where torch is faster than PyDOE.

Every Engineer dispatch must name the goal (M1–M4) it serves. New public functions are allowed this week (≤ 3 new root exports in total).

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **M1 Model features** | Gate first (independent review): locally linearized design for `y = f(x, θ) + ε`, homoscedastic noise, `M(θ₀) = Σ J(xᵢ)ᵀJ(xᵢ)` with `J = ∂f/∂θ` at a nominal `θ₀`; certificates apply to the linearized problem only and say so. Then a public `jacobian_features(model, X, *, params=None)` (all parameters, or a named subset such as the last layer; scalar output) whose rows feed `optimal_design` unchanged. Oracles: (a) a linear model gives exactly the existing polynomial feature rows (1e-12); (b) exponential decay `a·e^{−bx}` on [0, T], T ≥ 1/b: the 2-point local D-optimal design is {0, 1/b}; (c) Michaelis–Menten `Vx/(K+x)` on (0, x_max]: {K·x_max/(2K+x_max), x_max}. (b) and (c) were checked by the Architect by grid exhaustion in float64 (2001 and 4000 points) | none |
| **M2 Ask/tell** | A small public sequential API (name fixed in the gate; e.g. `next_batch(...)`) that proposes k new candidate IDs given already-run IDs, by augmentation through `fixed_ids`; the proposed IDs never repeat run IDs; with a fixed seed the loop is deterministic; for a linear model it matches `optimal_design(..., fixed_ids=...)` exactly | none |
| **M3 Simulated closed loop** | `examples/closed_loop_sim.py` (not in `src/`): a simulated noisy lab (Michaelis–Menten, seeded) plus a minimal backend protocol (`run(x) → y`), 4 rounds of ask → run → refit θ → ask, printing θ estimates; runs in CI's examples smoke in ≤ 60 s on CPU; one docs page "PyTorch model → experiments" with the MLP last-layer variant | none |
| **M4 Scale crossover** | Exactly the VISION.md P1.5 protocol: D, full quadratic, uniform random candidates in [−1,1]^k, k ∈ {3, 5}, n = 2p, m ∈ {10³, 10⁴, 10⁵}, per restart; PyDOE 1.6.0 fedorov / modified_fedorov / detmax vs TorchOED CPU vs TorchOED GPU, all in one Colab session, 30-min cap → TIMEOUT; report time, logdet, certified floor and the smallest m where TorchOED is faster at equal-or-better quality, or "no crossover found" | not run |

**Held-out additions (frozen in their Issue before M1 code merges):** two model-based cases not used anywhere else, e.g. an Emax model `E₀ + E_max·x/(ED₅₀+x)` (3 parameters) and a 2-compartment-like `a·e^{−bx} + c` (3 parameters) on fixed grids; reported at week end as-is.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards this week:** the M1 gate passes an independent fresh-context review before code; no claim that TorchOED beats active-learning methods; M4 is reported as-is, including a negative result; no tuning on held-out cases.

**Out of scope this week:** real robot or lab hardware, BoTorch/Ax adapters, Bayesian/minimax designs (P3), P2 industrial features, further A/I speed work beyond one background S task, real PyPI, Issue #13.
