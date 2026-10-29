# TorchOED — Algorithm Ecosystem Survey & Reuse-First Architecture Decisions

> **SAE V3.3 | Chief Architect | Research-only evidence memo | 2026-10-29**
>
> **Repository:** [JohnnyZPiggy/TorchOED](https://github.com/JohnnyZPiggy/TorchOED).
> **Existing contracts:** [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) and [SCIENTIFIC_MENTOR_RESEARCH.md](SCIENTIFIC_MENTOR_RESEARCH.md), especially the 91-name audit (chapters 17–24).
> **Engineering status:** The GitHub repo has three pre-existing files (LICENSE + two docs) before this memo, no README/package implementation and no PRs at time of review; M0 remains Issue [#1](https://github.com/JohnnyZPiggy/TorchOED/issues/1). This document adds **no dependency, code, PR, Issue or merge authorization**.
>
> **Review protocol:** Primary project docs / GitHub repository metadata / PyPI pages consulted on October 29, 2026. Maintenance activity and release numbers are **snapshots**, not continuous commitments. Licenses quoted only where confirmed by project metadata or official docs; inspect the pinned distribution license and NOTICE files before reuse. Evidence level: **SOURCE_REVIEW**, not local installation, comparative benchmarks or vetted runtime compatibility.

## 1. Executive architectural decision / 架构结论

**TorchOED is NOT an official PyDOE GPU port, not a wrapper around PyDOE, not a clone of BoTorch or PyOptEx, and not an all-in-one experiment-management platform.**

Its credible differentiation is a **small composable PyTorch-native numerical OED core**:

1. **Structured candidates / constructors**: classical factorial, response surface, mixture, LHS/QMC; distinguish integer combinatorics from tensor-linear-algebra workloads.
2. **Model/features and noise**: \`C[m,k] -> F[m,p]\`, differentiable \`J[m,p]\`, independent/heteroscedastic/dense-correlated noise with explicit replication semantics.
3. **Shared OED scoring**: D/A/E/G/I/C/V (plus S/T only with explicit mathematical semantics), matrix factorizations, weighted-inverse-trace kernels, rank/condition checks and batched candidate score reductions.
4. **Search and certificate**: exact (without replacement), replicated (with replacement), approximate weight designs, one-swap and only *genuine* k-swap/excursion when implemented, plus numerical and optimality-gap certificates.
5. **Optional ecosystem adapters** (future, NOT V1 core deps): BoTorch/GPyTorch for GP/BO; Pyro for Bayesian expected information gain; external constrained workflows where justified.

**Default runtime dependency stays \`torch\` only, as the pre-existing development plan mandates.** No PyDOE/SciPy/NumPy CPU computation inside the public hot path. Dev-only external CPU oracles are permissible in an isolated comparator environment. Any additional runtime dependency or change to original M0 scope requires a separate Architect decision and Scientist authorization.

The major performance hypothesis is not “Torch is faster than NumPy” but rather **Batched structured OED scoring can reuse factorizations and avoid O(n m) full matrix reconstructions, and on sufficiently large batches may have a GPU performance break-even.** No such speedup has been measured by this survey.

## 2. Confirmed landscape: first-order competitors, alternatives and reference oracles

Fields: **Mode** = proposed relationship to TorchOED (\`CORE\`, \`DEV_ORACLE\`, \`MATH_REFERENCE\`, \`OPTIONAL_LATER\`, \`AVOID_DEPENDENCY\`). **Maintenance** is a verified public GitHub snapshot when available. **License** means upstream project metadata, *not* an automatic permit to import/modify proprietary or third-party code.

| Project / verified source | Real capabilities / applicable TorchOED domains | Stack / license | Evidence, maintenance and decision |
|---|---|---|---|
| [PyTorch](https://pytorch.org/docs/stable/index.html), [SobolEngine](https://docs.pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html) | \`torch.linalg\`, \`bmm\`, \`vmap\`, autograd, \`cdist\`, \`quasirandom.SobolEngine\` (up to 21,201 dimensions in docs) | Native torch / upstream torch license | **CORE**. Reuse existing tensor and Sobol primitives; do not reimplement BLAS, Cholesky, Sobol direction tables or write CUDA in V1. The Sobol *generator* itself is not assumed CUDA-native; test generation vs tensor transfer separately. |
| [PyDOE](https://github.com/pydoe/pydoe), [fixed 1.6.0 source](https://github.com/pydoe/pydoe/tree/866d3f411942cb662af8054562dc1f7e07372401) | 91 root exports spanning factorial, mixture, response surface, QMC, space-filling, OED criteria/search, sequential, Taguchi, sensitivity | NumPy/SciPy; **BSD-3-Clause** | **DEV_ORACLE**, frozen 1.6.0 Git SHA \`866d3f411942cb662af8054562dc1f7e07372401\`. GitHub active (pushed 2026-10-28). Upstream installed-wheel parity was **BLOCKED** in mentor §24, not PASS. Several math/source semantic defects require triangulation, not blind copying. |
| [pyDOE3](https://github.com/relf/pyDOE3), [official status](https://pydoe3.readthedocs.io/) | Legacy PyDOE successor with overlapping functions | NumPy/SciPy; **BSD-3-Clause** | **AVOID_DEPENDENCY**: repo is archived and directs migration to main PyDOE. No need for parallel canonical oracles with same ancestry. |
| [PyOptEx](https://github.com/mborn1/pyoptex), [docs](https://pyoptex.readthedocs.io/en/latest/) | Optimal DoE, arbitrary criteria, mixture, split-plot/blocked design, budget/cost-optimal CODEX, experiment constraints | Python, predominantly CPU; **BSD-3-Clause** | **MATH_REFERENCE / possible external quality baseline**; current docs 1.2.1, GitHub updated 2026-10-28. Better reference than plain PyDOE for **fixed vs cost-constrained exact designs** and legal neighborhoods. Do not import its entire framework as TorchOED runtime. |
| [OApackage](https://github.com/eendebakpt/oapackage), [PyPI](https://pypi.org/project/oapackage/) | Orthogonal arrays, optimal/conference arrays, generalized wordlength, isomorphisms, D-efficiency | C++/Python; PyPI **BSD-3-Clause** | **DEV_ORACLE / specialist combinatorial reference**. GitHub active Oct 2026, but compiled stack may complicate environment; optional isolated test backend. Do not reproduce OA enumeration by GPU GEMM. |
| [SciPy QMC](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html) | Sobol, Halton, LHS including OA strength, Poisson disk, discrepancy metrics, scaling | NumPy/SciPy; standard SciPy open-source license | **DEV_ORACLE** for random/QMC criteria, not numeric runtime dependency. RNG/direction numbers/skip/roundup differ from PyTorch and PyDOE; test invariants and document sequence restrictions rather than claim all seeds match bitwise. |
| [QMCPy](https://github.com/QMCSoftware/qmcpy), [community](https://qmcsoftware.org/software/qmcpy.html) | QMC point generators, automatic measure transforms, integration, stopping tolerances | Python; **Apache-2.0** | **MATH_REFERENCE / optional advanced QMC oracle**, maintained Oct 2026; useful for quadrature correctness and error criteria beyond point count. |
| [OpenTURNS](https://openturns.github.io/openturns/latest/user_manual/designs_of_experiments.html) | Optimal LHS (annealing), space-filling scores, SmolyakExperiment and \`generateWithWeights\`, QMC, UQ | C++/Python; **license/reuse terms require file-level check** (GitHub metadata inconclusive) | **DEV_ORACLE** especially for *real weighted* sparse-grid and LHS optimization. It already implements permutation-update ideas; must compare mathematical objective definitions and numerical conditioning, not assume it is infallible. |
| [Pyomo.DoE](https://pyomo.readthedocs.io/en/stable/explanation/analysis/doe/doe.html) | Physics-based nonlinear model DoE, parameter uncertainty, Fisher information and constrained optimization | Pyomo algebraic models + solver stack; **license file check pending** | **MATH_REFERENCE / OPTIONAL_LATER**. Valuable for nonlinear OED formulations and solver-based baselines, not a tensor numerical dependency. |
| [pydex](https://github.com/KennedyKusumo/pydex) | Continuous model-based OED, pseudo-Bayesian/CVaR, D/A/E/V, Pyomo/IPOPT/DAE | Python/Pyomo/SciPy, **MIT** GitHub metadata | **MATH_REFERENCE** for continuous effort designs, nonlinear constraint handling; opt solvers are heavyweight. Maintained through Apr 2026, small user base. |
| [PyOED](https://web.cels.anl.gov/~aattia/pyoed/index.html) | Model-constrained OED, Bayesian inversion, sensor placement and data assimilation | Python research suite; [GitLab source](https://gitlab.com/ahmedattia/pyoed); **license not verified in this survey** | **MATH_REFERENCE / long-horizon** for robust/inverse-problem OED. Docs 2.1 under development; no claim about easy package installation or drop-in replacement. |
| [BoTorch](https://github.com/meta-pytorch/botorch), [dependency metadata](https://github.com/meta-pytorch/botorch/blob/main/pyproject.toml) | Batch GP Bayesian optimization, constraints, MC acquisition, differentiable BO | PyTorch+GPyTorch; **MIT** | **OPTIONAL_LATER** via adapter for sequential GP/BO, not a baseline runtime dependency. GitHub active 2026-10-28. Do **not** rebuild full GPytorch/BoTorch fit + acquisition framework without evidence of a gap. |
| [Pyro OED](https://docs.pyro.ai/en/dev/contrib.oed.html) | Bayesian OED expected-information-gain, Laplace, nested MC, variational methods | PyTorch probabilistic programming; **Apache-2.0** | **OPTIONAL_LATER** for EIG, which is mathematically distinct from local Fisher-logdet or GP expected improvement. Maintained Oct 2026. |
| [BoFire](https://experimental-design.github.io/bofire/), [repo](https://github.com/experimental-design/bofire) | Mixed-type constrained experimental domains, batch BO, chemical/material experimentation | Python, BO stack; **BSD-3-Clause** | **ARCHITECTURE_REFERENCE / OPTIONAL_LATER** for constraints/domain schema and user workflows; not a GPU OED core. GitHub active 2026-10-26. |
| [BayBE](https://github.com/emdgroup/baybe), [PyPI](https://pypi.org/project/baybe/) | Bayesian design of experiments with categorical/discrete/continuous search and constraints | Python, Torch ecosystem; **Apache-2.0** | **ARCHITECTURE_REFERENCE / OPTIONAL_LATER**, current repo active Oct 29 2026. Suitable workflow comparisons; not a reason to duplicate a complete optimization platform. |
| [SALib](https://github.com/SALib/SALib) | Sobol and Morris sensitivity analysis, FAST and other measures | Python; **MIT** | **DEV_ORACLE** for design sampling + sensitivity outputs. Note: a Sobol **sensitivity index** is distinct from a Sobol **QMC point sequence**. |
| [SMT](https://smt.readthedocs.io/en/stable/), [repo](https://github.com/SMTorg/smt) | Surrogate modeling, Kriging, sampling, benchmarks | Python; **BSD-3-Clause** | **DEV_ORACLE / OPTIONAL_LATER** for GP/surrogate tests and sampling domains, not a runtime dependency. |
| [discopt-doe](https://github.com/jkitchin/discopt-doe) | New JAX-based classical/mixture, Fisher-based D/A/E, model discrimination, sequential estimate–design workflows | JAX / GitHub **EPL-2.0** | **WATCH / architecture reference**. Published 2026 with few independent users; license file obligations and non-Torch stack discourage copying or production dependency. Its meaningful differentiator is model-identifiability integration, not merely JAX vectorization. |
| [NEXTorch](https://github.com/VlachosGroup/nextorch), [PyPI](https://pypi.org/project/nextorch/) | Older PyTorch+BoTorch experiment/BO toolkit | Older PyTorch/BoTorch constraints; **MIT** | **AVOID_DEPENDENCY**: last public Git push Aug 2024 and PyPI documents much older compatibility; current BoTorch is better forward reference. |
| [OptBayesExpt](https://github.com/usnistgov/optbayesexpt), [NIST](https://pages.nist.gov/optbayesexpt/index.html) | Sequential Bayesian experiment control and parameter learning | Python; license not independently checked here | **MATH_REFERENCE ONLY**, GitHub repository archived, not preferred live dependency. |

**Survey exclusions / other ecosystems:** \`dexpy\`, \`doepy\`, \`diversipy\`, \`pyLHD\`, \`DoEgen\`, \`IDAES PySMO sampling\`, \`OpenMDAO DOEDriver\`, \`UQpy\` and Optuna/Ax are informative but not essential first-order dependencies for a torch-native *mathematical* OED core. Their different niches are tracked in [the curated October 2026 ecosystem list](https://github.com/danieleongari/awesome-design-of-experiments). Do not assert lack of utility based only on repo popularity.

### 2.1 Maturity, source reliability and license gate

- **Actively maintained confirmed GitHub snapshots (not quality guarantees):** PyDOE (2026-10-28), PyOptEx (2026-10-28), OApackage (2026-10-13), BoTorch (2026-10-28), BoFire (2026-10-26), BayBE (2026-10-29), SciPy (2026), Pyro (2026-10-05), QMCPy (2026-10-26), OpenTURNS (2026-10-27), SALib (2026-10-29).
- **Archived / avoid as default:** pyDOE3 official GitHub archived. OptBayesExpt archived. NEXTorch last source push 2024 (not marked archived, but older compatibility). These are different risk classes.
- **Upstream known semantic risks:** frozen PyDOE has documented/source-vs-math discrepancies in [Mentor §§17–24](SCIENTIFIC_MENTOR_RESEARCH.md): strict sparse grid not genuine weighted Smolyak; Niederreiter degree-2 LFSR finite-rank first axis; Doehlert shell high-dim rank; modified Fedorov stale pool duplicate ID; DETMAX one-add/one-drop not a genuine multi-step excursion; criterion C/I/T/S different default semantics; Sobol skip/power2 fragility. Some are mathematical model/design limitations, not always bugs.
- **Licensing:** MIT project code is not permission to relicense copied BSD, Apache, EPL or other code as MIT. Add each source as a **citation/development reference**; a runtime dependency under its own license is distinct from copying code. Before any code reuse, review the exact tagged distribution LICENSE/NOTICE, attribution, patent and dependency obligations. This survey is not a legal clearance for any third-party code.

## 3. Direct Reuse Matrix: what is already solved, what TorchOED should actually build

| Capability | Reuse now / Oracle | Build or do not build | Architect decision |
|---|---|---|---|
| Dense matrix operations, autograd, batching | \`torch.linalg.cholesky_ex\`, \`solve\`, \`slogdet\`, \`eigvalsh\`, \`bmm\`, \`einsum\`, \`vmap\` | No custom CUDA/Triton/BLAS in V1 | **REUSE_CORE** |
| Sobol / basic space filling | \`torch.quasirandom.SobolEngine\`; SciPy qmc/OApackage/OpenTURNS reference | Thin torch-native interface around existing engine plus explicit skip, bounds, device, RNG contracts; own LHS permutations as warranted | **REUSE_CORE + SPECIALIZED_BUILD** |
| Full/fractional factorial, BB/CCD, mixture | PyDOE docs, OApackage, PyOptEx | Integer/broadcast constructors **only as needed**; model-rank and alias checks independent of names. No giant suite before OED core | **BUILD_LATER** |
| Information / Fisher / noise | torch weighted matrix ops, solves, autograd Jacobian; Pyomo.DoE/pydex reference | Shared \`F\`, \`J\`, covariance semantics, approximate vs exact normalization, with/without replacement and missing-rank diagnostics | **PRIMARY_DIFFERENTIATOR** |
| D/A/E/G/I/C/V OED scoring | torch.linalg + independently derived closed forms | Shared batched scoring, rank/condition reporting, exact semantics. A/I/C/V weighted inverse-trace family, D logdet, E eigspace, G max variance | **PRIMARY_DIFFERENTIATOR** |
| Exact Fedorov / k-exchange | PyOptEx exchange structure, OApackage/real full-swap enumeration, mentor CPU oracles | Candidate-ID strict masks, tiled incremental rank-two, verified acceptance and optional true multi-swap later | **PRIMARY_DIFFERENTIATOR** |
| Approximate weights and quality certificates | Mentor KKT/equivalence derivations; Pyomo models or SciPy optimize external dev oracles | constrained simplex weights and exact-design lower bounds as later mathematical modules; do not promise global exact optimization | **LATER_CORE** |
| True Smolyak and special QMC | OpenTURNS SmolyakExperiment, QMCPy, symbolic polynomial oracle | Scope explicit 1D quadrature family, weights, sparse multi-index and integration exactness. **Do not port PyDOE's misleading sparse grid** | **DEFER_UNTIL_ORACLES** |
| GP + Bayesian optimization | BoTorch/GPyTorch; BoFire/BayBE workflow references | Optional adapter; no redundant GP kernel/BO product in V1 | **OPTIONAL_ADAPTER** |
| Bayesian EIG | Pyro OED | Optional compatibility bridge; not a D-opt/Fisher alias | **FUTURE_ONLY** |
| Sensitivity / Morris / Saltelli | SALib + Torch QMC | small sampling helpers only if user-facing demand exists; no full SALib clone | **REFERENCE_ONLY** |
| Mixed categorical / cost/constraint domain | PyOptEx, BoFire, BayBE | lightweight \`candidate_id\`, factor schema and feasibility mask; avoid huge product framework | **REFERENCE_ONLY** |

## 4. Lean architecture map and explicit contracts

Proposed composition (architecture diagram, **NOT code layout or M0 deliverables**):

\`\`\`text
Classic DOE / LHS / Sobol / external provided candidates
                          |
           CandidateSet {coordinates, ids, constraints}
                          |
        FeatureMap / Jacobian {F[m,p], J[m,p,...]}
                          |
                NoiseModel / Replication
                          |
   InformationMatrix {M[p,p], normalization, SPD/rank status}
                          |
           Criterion {score, validity, errors}
                          |
   +----------------------+------------------------+
   |                      |                        |
Exact selected IDs    Approx. simplex w      Sequential/adaptive
one/k-exchange       KKT / dual bound       optional BoTorch/Pyro
   |                      |                        |
   +----------------------v------------------------+
          independent CPU oracle + GPU break-even
\`\`\`

**Proposed invariant types (NOT M0 implementation):**
- \`candidate_id\` identity, physical coordinates, and feature rows remain separate; numeric coordinate \`isclose\` never stands for candidate-ID uniqueness.
- \`Selection\` has \`allow_repeats\`, or separate \`counts\` vs \`unique_ids\`. With correlated observation covariance, repeated event identity/covariance must be explicit (not a repeated index into one observed event).
- \`InfoMatrix\` carries \`normalized\` and exact \`n\`; distinct design sizes cannot compare raw determinant scores without scale conversion.
- \`CriterionSpec\` carries objective sign, *fixed* moment H, contrast c or evaluation set, and expected units; E-opt multiple eigenvalues need eigenspace PSD certificate, S score is not correlation determinant, T projection vs residual model-discrimination is a real modeling choice.
- Validity is reported: \`rank\`, Cholesky failure, condition diagnostics, regularization used/forbidden, threshold/tolerance and unsafe float32 near singularity.
- Output to caller should include candidate index sequence (or counts), score, validity and diagnostics. Determinism is per engine/device, not across SciPy vs torch RNG streams.

**Avoid overdesigning the M0 spine:** none of these dataclasses/interfaces are M0 requirements; prefer a tiny torch-only package + evidence ledger in the existing [M0 issue](https://github.com/JohnnyZPiggy/TorchOED/issues/1), then freeze only proven API contracts in M1/M2.

## 5. Prioritization and experimental hypotheses (no benchmark fabricated)

### Experiment H1 — batched criterion reuse

For m candidates, n selected, p parameters, B independent problem instances:
- full Gram recompute per trial costs roughly O(np² + p³), repeated O(n(m−n)) trials;
- share current M factorization, derive rank-two Woodbury/determinant ratios and weighted A/I/C/V scores, exploit tile/small matrix solves and GPU batched GEMM;
- benchmark **PyDOE CPU, Torch CPU full-recompute, Torch CPU rank-update, Torch CUDA rank-update** with equal criterion, quality, candidate IDs, stop and fairness protocol. Do not claim speedup until synchronized wall time and peak memory measured. Fail/reject singular and borderline swaps rather than silently regularize.

### Experiment H2 — mixture and constrained design correctness

Compare Torch-generated small candidate sets to PyOptEx, OApackage, PyDOE and independent exhaustive oracles, with **model basis specified**. Assert actual design matrix rank/alias class; for four-factor small composite full quadratic, 20 runs may yield rank 12<15. Verify without/with-replacement separately.

### Experiment H3 — dense-noise Fisher cost

Contrast independent variance GEMM with full covariance selected-submatrix factorization; benchmark batch/candidate choices, covariance diagonal vs dense vs repeated measurement. Mathematical equality of slicing inverse covariance must be checked (generally false). No CUDAGPU advantage assumed for small or irregular instances.

### Experiment H4 — Smolyak rule

Separate point-cloud outputs from **quadrature with weights**. Check node nesting, negative combination weights, unity weight sum, and monomial integration accuracy against independent rational 1D rules / OpenTURNS; only then consider tensor evaluation.

### Promotion gate / cuOED

Promote a CUDA-native custom kernel **only** if a live GPU trace demonstrates that torch-native batched OED remains materially bottlenecked on a well-defined operation, while CPU/torch alternatives do not meet a specified workload target. Record correctness, transfers, launch overhead, synchronization and GPU memory. This does not authorize cuOED now.

## 6. Decision record / next engineering handoff

| ADR | Status | Decision |
|---|---|---|
| ADR-01 | **ACCEPT** | PyDOE frozen 1.6.0 is dev-only source comparator, NOT runtime backend. Real-wheel parity from mentor §24 is BLOCKED, so do not pretend installed-PyDOE tests passed. |
| ADR-02 | **ACCEPT** | \`torch\` sole default numerical runtime; torch.linalg and existing Sobol engine first. No new dependencies or PyTorch-only slogan unsupported by code. |
| ADR-03 | **ACCEPT** | Numerical OED core (model → information → criterion → exact search) before trying to match all 91 public PyDOE names. |
| ADR-04 | **ACCEPT** | Candidate IDs/replication, normalization, information-matrix rank and explicit criterion defaults are first-class mathematical contracts; no silent quirks copied from PyDOE. |
| ADR-05 | **ACCEPT** | PyOptEx, OApackage, OpenTURNS, SciPy QMC, Pyomo.DoE serve as **independent external math/structure oracles** in optional dev setups. No package new default dep. |
| ADR-06 | **DEFER** | BoTorch, Pyro, BoFire, BayBE, PyOED adapters; optional only after narrow user-facing demand and compatibility/license assessment. |
| ADR-07 | **DEFER** | Custom CUDA/cuOED; measurable break-even/roofline plus correctness gate first. |
| ADR-08 | **PENDING** | Exact vs statistical parity for all PyDOE edge cases; isolated wheel acquisition remains blocked in current CPU environment. |
| ADR-09 | **PENDING** | Actual runtime compatibility matrix Python/PyTorch/CUDA, license and third-party notices per supported release. No package was installed by this survey. |

**Immediate Architect handoff to the existing M0 Engineer, with NO scope expansion:** add pointers to this ecosystem memo when preparing the 91-export coverage inventory; use torch-native primitives as candidates for reuse; keep PyDOE an optional oracle; do not install all these external projects as dependencies, copy code, build GPU kernels or widen Issue #1. Engineer only needs the pre-authorized minimal spine, CPU smoke and honest evidence.

**Next *research* task, outside current M0:** choose 3 or 4 external reference packages (SciPy, PyOptEx, OApackage, OpenTURNS), freeze versions/inputs/criterion definitions and execute bounded *actual-package* CPU probes in a network-capable existing environment. Pair them with native torch CPU mathematical checks; then plan the M1/M2 feature-level build after Scientist/Architect decision. Previous PyDOE wheel DNS blockade is an environmental dependency, not an argument that PyDOE lacks a release.

## 7. Source index (primary sources)

- **Baseline / tensor**: [PyDOE GitHub](https://github.com/pydoe/pydoe); [fixed 1.6.0](https://github.com/pydoe/pydoe/tree/866d3f411942cb662af8054562dc1f7e07372401); [torch Sobol](https://docs.pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html); [PyDOE3 maintenance notice](https://pydoe3.readthedocs.io/).
- **Classical optimal / combinatorial**: [PyOptEx docs](https://pyoptex.readthedocs.io/en/latest/); [PyOptEx repo](https://github.com/mborn1/pyoptex); [OApackage PyPI](https://pypi.org/project/oapackage/); [Pyomo.DoE docs](https://pyomo.readthedocs.io/en/stable/explanation/analysis/doe/doe.html); [pydex repo](https://github.com/KennedyKusumo/pydex); [PyOED official docs](https://web.cels.anl.gov/~aattia/pyoed/index.html).
- **Sampling / integration / UQ**: [SciPy QMC](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html); [QMCPy GitHub](https://github.com/QMCSoftware/qmcpy); [OpenTURNS designs](https://openturns.github.io/openturns/latest/user_manual/designs_of_experiments.html); [SALib](https://github.com/SALib/SALib); [SMT](https://smt.readthedocs.io/en/stable/).
- **Bayesian / adaptive**: [BoTorch](https://github.com/meta-pytorch/botorch); [BoTorch pyproject](https://github.com/meta-pytorch/botorch/blob/main/pyproject.toml); [Pyro OED](https://docs.pyro.ai/en/dev/contrib.oed.html); [BoFire](https://experimental-design.github.io/bofire/); [BayBE](https://github.com/emdgroup/baybe).
- **Watch/legacy**: [discopt-doe](https://github.com/jkitchin/discopt-doe); [NEXTorch](https://github.com/VlachosGroup/nextorch); [OptBayesExpt](https://github.com/usnistgov/optbayesexpt); [curated survey (2026-10)](https://github.com/danieleongari/awesome-design-of-experiments).

**Survey status:** \`ARCHITECT_REUSE_FIRST_SURVEY_COMPLETE\`; \`PACKAGE_INTEGRATION_NOT_RUN\`; \`TORCHOED_M0_UNCHANGED\`; \`CUDA_NOT_RUN\`.