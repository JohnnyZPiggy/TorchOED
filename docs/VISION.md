# TorchOED vision (Scientist direction, 2026-11-01; draft v1 — the Scientist edits this file, the loop does not)

## North star
**One PyTorch-native engine for the whole of optimal experimental design** — classical exact designs, industrial DOE, nonlinear and Bayesian model-based designs — that matches or beats PyDOE, R (AlgDesign, OptimalDesign, skpr, acebayes, ICAOD, FrF2/DoE.base, rsm) and JMP's Custom Designer on quality, adds what none of them combine, and scales on GPU.

Today these capabilities are spread over a dozen tools with different languages, criteria definitions and no quality guarantees. TorchOED unifies them under one tensor contract.

## Four signatures (what makes it more than "another DOE package")
1. **Certified designs everywhere.** Every returned design carries a proved lower bound on its efficiency (Kiefer–Wolfowitz/equivalence-theorem duality for D, A, I and compound criteria), not just a number.
2. **Autograd-native model-based design.** Users write a model `f(x, θ)` in torch; Fisher information comes from Jacobians, so nonlinear local, Bayesian (prior-averaged) and robust/minimax designs need no hand-derived formulas.
3. **GPU scale.** Batched low-rank exchange over 10⁵–10⁶ candidates, many restarts and Monte-Carlo prior draws in parallel, with end-to-end measured (never assumed) speed.
4. **Differentiable and composable.** Continuous factors optimized by gradients, designs as tensors that plug into BoTorch/Ax/active-learning loops.

## Capability map (target 1.0) and phases
| Phase | Theme | Capabilities | Exit evidence |
|---|---|---|---|
| **P1** (weeks 1–3) | Unified exact-design engine | criterion-generic exchange (D, A, I) via shared low-rank identities; augmentation (fixed runs); linear/disallowed-combination constraints; categorical factors; certificates for each criterion; PyPI 0.6, docs site, typed API | held-out benchmark vs PyDOE + R AlgDesign: quality ≥, time ≤ 3× fastest comparable |
| **P2** (weeks 4–6) | Industrial DOE parity | split-plot / hard-to-change factors, blocking, mixture and process variables, covariates, alias-aware and compound criteria, definitive screening designs, design diagnostics (correlation map, FDS, power) | case-by-case match of JMP/skpr reference designs on published examples |
| **P3** (weeks 7–10) | Torch-native frontier | nonlinear local designs via autograd; Bayesian/pseudo-Bayesian designs by batched prior Monte Carlo; minimax/robust designs; approximate (continuous-weight) designs on GPU with rounding + certificates; GPU-scale exchange | reproduce published nonlinear/Bayesian optimal designs (ICAOD/acebayes examples); same-machine CPU vs GPU curves |
| **P4** | Ecosystem, 1.0, research | BoTorch/Ax adapters for sequential design; gradient-based continuous design; 1.0 with semver and deprecation policy; JOSS paper; methods paper only if a Mentor literature review confirms novelty | external users, issues fixed, JOSS submission |

## Guardrails
- **Library, not GUI.** JMP's interactive platform is out of scope; parity means design capability and quality, not interface.
- **No claim without evidence.** "Beats X" only on a frozen public benchmark; "nobody else does Y" only after a Mentor landscape review.
- **Held-out benchmark.** A case set the loop never optimizes against is run once per week and reported as-is.
- **Independent math review.** Every new gate document is attacked by a reviewer without the author's context (fresh subagent or the Mentor) before it is merged.
- **Weekly Scientist check.** The Scientist reads the weekly report and redirects; the loop executes, it does not choose the destination.
