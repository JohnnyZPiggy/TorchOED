# TorchOED roadmap (detailed plan, Scientist decision 2026-11-28)

This file details the phases in [`VISION.md`](VISION.md). If the two ever disagree, `VISION.md` wins. The Architect drafts each `docs/WEEK_GOAL.md` from the current phase section below. At the start of each phase it adds that phase's capability rows to [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md). Like `VISION.md`, this file changes only by committing a Scientist decision.

Decision of 2026-11-28 (in chat with the Architect: "全加，P9 提前，同时把新的整体研发计划全面细化下"): add P7–P10 and the infrastructure block I. Order after 1.0: **P5 → I → P9 → P7 → P6 → P8 → P10**.

## 1. Order and week blocks

Weeks are goal weeks; they advance when their goals are met (VISION.md, weekly cadence). Every 4th week is a consolidation week (C).

| Weeks | Block | Theme |
|---|---|---|
| 19 | P2.5 (end) | Studio analysis, REML prediction, held-out H52–H54 |
| 20 | C | release 0.x; JOSS wording → technical report in README/website |
| 21–23, 24 C | **P3** | torch-native frontier (approximate, nonlinear, Bayesian, minimax, 10⁶ scale) |
| 25–27, 28 C | **P4** | ecosystem and **1.0**; technical report v1 (arXiv); hosted Studio |
| 29–31, 32 C | **P5** | cuOED GPU backend |
| 33–35, 36 C | **I** | OEDBench + certified design catalogue, MCP server, R bridge |
| 37–39, 40 C, 41–43, 44 C | **P9** | robust and game-theoretic design |
| 45–47, 48 C, 49–51, 52 C | **P7** | Bayesian and adaptive design with neural estimators |
| 53–55, 56 C | **P6** | TorchOED Mobile |
| 57–59, 60 C, 61–63, 64 C | **P8** | inverse problems and dynamical systems |
| 65–67, 68 C, 69–71, 72 C | **P10** | experimentation platform and ML data selection |

## 2. Rules for every post-P2.5 phase

1. **Phase entry.** A Mentor landscape review comes first: what exists, what is validated against what, and which reference implementations can serve as oracles. Then a gate document goes through independent math review. Then matrix rows are added. No feature code starts before its gate.
2. **Packaging.** Each new direction lives behind an optional extra (`torchoed[cuda]`, `torchoed[bayes]`, `torchoed[inverse]`, `torchoed[platform]`) or a separate package (`torchoed-studio`, `torchoed-mcp`, the R package). The core library's runtime dependency stays torch only.
3. **Evidence.** Every numerical claim needs an exact small oracle (closed form, Fraction or enumeration). Every capability needs an external published reference where one exists (R via `r-oracle.yml`; otherwise NOT_RUN, recorded). Each week adds 2–4 frozen held-out cases. GPU numbers are same-host only.
4. **Signature.** Each phase exit shows at least one of the four signatures on a published example: certificate, measured GPU speed-up at equal quality, autograd/neural model, or grid-free differentiable design.
5. **Size.** Weeks are sized for 2 Engineer tasks in flight plus the GPU track (Scientist decision 2026-11-26). Each week has at most 3 goals.
6. **Technical report.** Every minor release updates `docs/TECH_REPORT.md`, the arXiv source from P4 on: methods, certificates, oracle evidence, the held-out record and GPU curves. There is no journal or conference submission until the Scientist calls the project complete, and then only to a venue with no publication fee (VISION.md, Publication).

## 3. Phase details

### P3: torch-native frontier (weeks 21–23)
- **W21 approximate designs.** Continuous-weight designs on a candidate set via multiplicative, vertex-exchange and REX-type algorithms, batched on GPU. Certificates come from the equivalence theorem: D needs max sensitivity ≤ p; A and I use the analogous bounds. Efficient rounding to n runs (Pukelsheim–Rieder) with the rounding loss reported. One internal backend interface routes search-heavy kernels (groundwork for P5). Oracle: R `OptimalDesign` (REX) on published examples.
- **W22 nonlinear and Bayesian.** Local designs for user `f(x, θ)` via autograd Jacobians: Michaelis–Menten, Emax, exponential decay, logistic. Pseudo-Bayesian D/A/I by batched prior Monte Carlo, with MC standard errors. Oracles: closed-form local designs from the literature, plus `ICAOD` and `acebayes` published examples.
- **W23 minimax and scale.** Standardized maximin-efficiency designs over parameter sets, solved as approximate designs with a duality-gap certificate. GPU exchange to 10⁶ candidates using the Mentor #125 pair-screening bound. Same-host CPU vs GPU curves on every GPU type obtained.
- **Exit.** The published nonlinear, Bayesian and minimax designs are reproduced. Same-host curves are recorded.

### P4: ecosystem and 1.0 (weeks 25–27; release in week 28)
- BoTorch/Ax adapters for sequential batches (`next_batch` ↔ Ax trials), grid-free gradient-based continuous design, discrete choice designs (`idefix` oracle), incomplete block and crossover designs (`agricolae` / `crossdes` oracles).
- API freeze, semver and deprecation policy, migration notes.
- Week 28: **1.0** on PyPI (needs the Scientist's token). Technical report v1 goes to arXiv. Studio is hosted on Hugging Face Spaces (free tier; Scientist decision 2026-11-28, 2A).
- **Exit.** Every P4 row is complete. 1.0 is released. Report v1 is public.

### P5: cuOED (weeks 29–31)
- Profile first; build only what the profile shows. Triton kernels for batched exchange scoring, low-rank updates and certificate evaluation; raw CUDA only where Triton is measured to be the limit.
- GPU first-order solvers for approximate designs: Frank–Wolfe with away steps, and mirror descent on the simplex. Where the problem is an LP or conic program (e.g. discretized minimax, E-optimality), use PDHG-type solvers in the style of cuPDLP / CuClarabel, through thin optional adapters if a maintained package exists (Reuse before build).
- FP32/TF32 search; every reported certificate re-verified in float64.
- **Exit.** Same-host CPU vs PyTorch-GPU vs cuOED curves up to 10⁶ (10⁷ if memory allows). Designs and certificates equal to the float64 CPU path within stated tolerances.

### I: infrastructure (weeks 33–35)
- **I1 OEDBench and certified design catalogue.** A frozen problem suite with a JSON problem schema (model, region, constraints, n, criterion). Runners for TorchOED and for the R packages via `r-oracle.yml`. A results page. The catalogue holds the best known design for standard problems (factorial and response-surface models for k ≤ 8 on cube, sphere and simplex regions; D, A and I), each with its certified efficiency lower bound. Released with a Zenodo DOI (free).
- **I2 MCP server.** A separate package, `torchoed-mcp`: tools for generate, evaluate, compare, fit and profile. Stateless, calling public API only, tested against direct library calls.
- **I3 R bridge.** An R package `torchoed` via `reticulate`: thin wrappers returning data frames, tested on `r-oracle.yml`. Distributed through GitHub/r-universe (free); a CRAN submission needs a Scientist decision.
- **Exit.** The catalogue covers ≥ 100 problems with certificates. MCP and R bridge run the published Studio examples end to end.

### P9: robust and game-theoretic design (weeks 37–39 and 41–43)
- **Entry.** A Mentor landscape review covering minimax and maximin-efficiency OED, model-robust designs, distributionally robust OED, and minimax-regret designs, with a novelty check for each item below.
- **Capabilities.**
  - Maximin efficiency over parameter boxes and finite model sets, with general-equivalence-theorem certificates for minimax criteria (statement fixed in the gate).
  - Model-robust compound and maximin designs across candidate models.
  - **Relative (minimax-regret) robust designs**: maximize the worst-case ratio to the per-scenario optimum, the design analogue of relatively robust decisions.
  - Distributionally robust Bayesian designs: worst-case expected efficiency or EIG over prior ambiguity sets (moment, φ-divergence, Wasserstein).
  - **Game view**: design vs nature as a zero-sum game. Nature's mixed strategy is the least favourable prior. Saddle points are computed by GPU first-order methods (extragradient / PDHG), with duality-gap certificates.
- **Oracles.** Closed-form minimax designs from the literature, `ICAOD` minimax examples, and exact small cases by LP or enumeration.
- **Exit.** Published minimax designs reproduced, with certificates. If the Mentor review confirms novelty, a methods-paper draft goes to the Scientist, who decides whether to write it.

### P7: Bayesian and adaptive design with neural estimators (weeks 45–47 and 49–51)
- **Entry.** Reuse check of Pyro's `contrib.oed` and other maintained EIG libraries (adapter vs build).
- **Capabilities.**
  - EIG estimators: nested Monte Carlo, prior contrastive estimation, and variational posterior/marginal bounds. Each reports its bias direction and MC standard error.
  - Gradient-based static Bayesian designs via reparameterization; implicit (likelihood-free) models.
  - Amortized sequential design: policy networks in the Deep Adaptive Design family, trained on GPU, deployed through `next_batch`.
- **Oracles.** Linear-Gaussian EIG in closed form, ½·log det(I + σ⁻² Σ_θ^{1/2} XᵀX Σ_θ^{1/2}). Published DAD benchmarks (e.g. location finding) reproduced within the reported spread. `acebayes` examples.
- **Exit.** Closed-form EIG matched within stated MC error. One published adaptive benchmark reproduced. Package `torchoed[bayes]`.

### P6: TorchOED Mobile (weeks 53–55)
- Android and iOS apps from Studio, as one codebase (e.g. Capacitor). Features: offline replay, connecting to a Studio server or the user's machine, viewing and running a design, entering responses, fit and profiler.
- Automated workflow tests in mobile emulation.
- App-store accounts and fees: Scientist decision.
- **Exit.** The published Studio examples run end to end on both platforms.

### P8: inverse problems and dynamical systems (weeks 57–59 and 61–63)
- **Capabilities.**
  - **Sensor placement for linear Bayesian inverse problems**: A- and D-optimal designs with randomized trace and log-det estimators (Hutchinson, Lanczos) and adjoint gradients by autograd. Sparsity-relaxed weights, then rounding, with the rounding loss reported. Scales to 10⁶ parameters on GPU.
  - A pure-torch finite-difference Poisson / advection–diffusion example (no FEniCS dependency).
  - **Dynamical systems**: optimal sampling times for ODE models (`torchdiffeq` as an optional extra), and input design for linear state-space identification (frequency-domain Fisher information), including identification-for-control criteria.
  - Population PK/PD sampling designs.
- **Oracles.** Dense exact small inverse problems, and `PopED` published examples via `r-oracle.yml`.
- **Exit.** A PopED example matched. A 10⁶-parameter sensor placement with a certified bound checked against a dense small instance. Package `torchoed[inverse]`.

### P10: experimentation platform and ML data selection (weeks 65–67 and 69–71)
- **Online experiments.**
  - Power and sample size for means and proportions.
  - CUPED / regression adjustment.
  - Stratified and blocked randomization.
  - Group-sequential designs with alpha spending.
  - Always-valid sequential tests.
- **ML data selection.**
  - D/V-optimal subset selection on embeddings.
  - Determinant-based batch selection.
  - Evaluation-set selection that estimates a model's score from n items with a stated standard error.
- **Oracles.** R `pwr` and `gsDesign` examples; exact enumeration on small selection problems; honest baselines (random, k-center), measured only.
- **Exit.** The `pwr` and `gsDesign` examples are matched. A frozen data-selection benchmark is reported as-is. Package `torchoed[platform]`.

## 4. Scientist decision points (collected)

| When | Decision |
|---|---|
| W28 | PyPI token for 1.0; arXiv submission of technical report v1 |
| I3 | CRAN submission or r-universe only |
| P9 exit | whether to write a methods paper |
| P7 entry | Pyro adapter vs own implementation (after the Mentor reuse check) |
| P6 | app-store accounts and fees |
| any time | declaring the project complete; choosing a no-fee venue |

## 5. Risks and how they are handled

- **Breadth over depth.** Each phase closes only when its matrix rows are complete (Completeness rule). Unfinished rows carry over before new work starts.
- **Thin oracles in research phases (P7–P9).** Without an external reference, results are reported as OBSERVED with exact small checks, never as "beats X".
- **GPU allocation.** Any GPU, any time; NOT_RUN is recorded and retried (Scientist decisions 2026-11-26).
- **Engineer usage.** Weeks are sized for 2 tasks in flight. The Scientist can lower the batch or pause at any time.
- **Repository growth.** New directions live in extras or separate packages. Consolidation weeks prune.
