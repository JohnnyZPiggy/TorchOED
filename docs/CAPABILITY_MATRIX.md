# Capability matrix (Completeness rule and parity checklist, VISION.md v3; baseline `main@4841b68`, 2026-11-16)

Every feature PR updates its row. Cell legend: ✅ done with evidence · ◐ partial (say what is missing) · ❌ missing · N/A not applicable (reason) · NR external reference NOT_RUN. Columns follow the Completeness rule: **Crit** D/A/I · **Aug** `fixed_ids` · **Con** candidate constraints / categorical coding · **Cert** efficiency certificate · **Diag** `design_diagnostics` incl. power · **Dev** CPU float64 / measured GPU · **Ext** external published reference · **Docs** API + guide snippet · **HO** held-out case.

## P1 / P1.5 — engine and model-driven design
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| `optimal_design` (plain exact) | ✅ D A I | ✅ | ✅ | ✅ KW dual floor | ✅ | ✅ CPU; GPU measured for D only (`w5-crossover`) | ✅ PyDOE, AlgDesign | ✅ | ✅ H1–H8 | done; A/I GPU timing → W19 |
| `jacobian_features` + `next_batch` | ✅ D A I | ✅ | ✅ | ✅ linearized only | ✅ | ◐ CPU only measured | ❌ no published nonlinear design reproduced | ✅ | ✅ H9–H10 | W17 (published nonlinear examples) |
| criterion functions | ◐ D A E c evaluated; G, V, I-as-function missing | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | W11 |

## P2 — industrial DOE (completion weeks 9–11)
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| blocking `blocks=` | ✅ D A I (#228) | ❌ | ✅ | ✅ dual efficiency floor vs unblocked approximate optimum (#236) | ✅ M_B SPV/VIF/power (#234) | ◐ CPU float64 only | ✅ AlgDesign `optBlock` (H11) | ✅ | ✅ H11 | **W9** |
| split-plot `whole_plots=` | ✅ D A I (#232) | ❌ | ✅ | ✅ dual efficiency floor vs OLS approximate optimum, conservative (#236) | ✅ GLS SPV/VIF; power exact (balanced orthogonal) else containment-df approx. (#234) | ◐ CPU float64 only | ✅ skpr `gen_design` (`w8-skpr`) | ✅ | ✅ H14 | **W9** |
| blocks + split-plot together | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | W9 if the gate covers it, else W11 |
| definitive screening | N/A constructor | N/A | N/A | N/A (exact orthogonality proved) | ✅ | ✅ | ◐ Paley orders only (q prime); other k raise | ✅ | ✅ H12 | W10 (non-Paley conference matrices) |
| fractional factorial by generators / resolution, fold-over, alias matrix | ❌ | ❌ | ❌ | N/A | ❌ alias diagnostics | ❌ | ❌ FrF2 | ❌ | ❌ | **W10** |
| Plackett–Burman, central composite (incl. blocked), Box–Behnken | ❌ | N/A | N/A | N/A | ❌ | ❌ | ❌ rsm / published tables | ❌ | ❌ | **W10** |
| mixtures: simplex lattice | N/A constructor | N/A | N/A | N/A | ✅ | ✅ | ❌ | ✅ | ✅ H15 | done |
| mixtures: simplex centroid, constrained (bounds, extreme vertices), optimal mixture D/A/I with Scheffé models | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ JMP/mixexp examples | ◐ manual Scheffé column selection documented | ❌ | **W11** |
| mixture-process variables | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **W11** |
| covariates (fixed per-run values) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **W11** |
| alias-aware and compound (e.g. D+I) criteria | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ JMP alias-optimal | ❌ | ❌ | **W11** |
| power | ◐ per-coefficient t-test, OLS, σ = 1 | N/A | N/A | N/A | ◐ blocked + split-plot power ✅ (#234); no effect-group F-test | ✅ | ✅ SciPy (H16) | ✅ | ✅ H16 | **W9** (structured), W11 (F-test) |
| space filling: random LHS, Sobol | N/A | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | done; maximin/MaxPro LHS → backlog |

## P3 — torch-native frontier (weeks 17–19)
| Capability | Status | Week |
|---|---|---|
| approximate designs (multiplicative / Frank–Wolfe weights) on GPU + equivalence-theorem certificates + efficient rounding | ❌ | W17 |
| nonlinear local designs reproduced from the literature (ICAOD examples) | ❌ | W17 |
| Bayesian / pseudo-Bayesian designs by batched prior Monte Carlo | ❌ | W18 |
| minimax / maximin-efficiency robust designs | ❌ | W19 |
| GPU-scale exchange to 10⁶ candidates | ◐ 10⁵ measured for D | W19 |

## P4 — ecosystem and 1.0 (weeks 21+)
| Capability | Status | Week |
|---|---|---|
| BoTorch/Ax adapter for sequential design | ❌ | W21 |
| gradient-based continuous design | ❌ | W21–22 |
| API freeze, semver, deprecation policy, 1.0 on PyPI | ❌ (TestPyPI 0.7.0) | W23–24 |
| discrete choice designs (idefix-style) | ❌ | W22 |
| JOSS paper | ❌ | W24 |

## Parity checklist vs JMP 18 DOE and R (Scientist 2026-11-16)
One row per capability a practitioner would look for in JMP or the named R packages. "Advantage" names the TorchOED signature to demonstrate on that row where one applies.

| Reference capability | JMP | R reference | TorchOED row / status | Advantage to show | Week |
|---|---|---|---|---|---|
| Custom design, D/A/I, constraints, disallowed combinations, categorical | Custom Design | AlgDesign `optFederov` | ✅ `optimal_design` | certificate; GPU at 10⁵ | done |
| Augment design (add runs to existing) | Augment Design | AlgDesign `augment`-style via fixed rows | ✅ plain; ◐ structured (W9-B3) | certificate on the augmented design | W9 |
| Blocking, fixed and random blocks | Custom Design | AlgDesign `optBlock`, skpr | ✅ D A I (#228); Cert ✅ (#236); Aug pending (W9-B3) | certificate | W9 |
| Split-plot, hard-to-change factors | Custom Design | skpr `gen_design` | ✅ D A I (#232); Cert ✅ (#236); Aug pending (W9-B3) | certificate; GPU | W9 |
| Split-split-plot, strip-plot (very-hard-to-change) | Custom Design | skpr | ❌ | — | W15 |
| Evaluate Design: power, prediction variance, FDS, correlation map, efficiencies | Evaluate Design | skpr `eval_design`, `Vdgraph` | ◐ unstructured only (W9-B4) | — | W9 |
| Alias matrix, alias-optimal and compound criteria | Custom Design | FrF2 alias tools | ❌ | — | W10–W11 |
| Power by simulation for GLMs (logistic, Poisson) | Sample Size Explorers | skpr `eval_design_mc` | ❌ | GPU Monte Carlo | W15 |
| Sample size explorer (power vs n) | Sample Size Explorers | `pwr`, skpr | ◐ power at fixed n | batched over n | W15 |
| Compare Designs (side-by-side efficiencies, power, FDS) | Compare Designs | — | ❌ | — | W15 |
| "If Possible" terms, Bayesian D/I (DuMouchel–Jones) | Custom Design | — | ❌ | certificate | W14 |
| Definitive screening | Definitive Screening | `daewr` DSD | ◐ Paley orders only | exact orthogonality proof | W10 |
| Fractional factorial by generators / resolution, minimum aberration, blocked FF, fold-over | Screening | FrF2, DoE.base | ❌ | — | W10 (MA, blocked FF → W15) |
| Plackett–Burman | Screening | FrF2 `pb` | ❌ | — | W10 |
| Group-orthogonal supersaturated designs | Screening | — | ❌ | — | backlog |
| Central composite (incl. blocked), Box–Behnken | Response Surface | `rsm` | ❌ | — | W10 |
| Mixture: simplex lattice / centroid, extreme vertices, ABCD, constrained, mixture-process | Mixture Design | `mixexp` | ◐ simplex lattice only | optimal mixture with certificate | W11 |
| Covariates (incl. hard-to-change covariates) | Custom Design | AlgDesign (fixed rows) | ❌ | — | W11 |
| Space filling: LHS (maximin), MaxPro, uniform, sphere packing, fast flexible filling (constrained) | Space Filling Design | `lhs`, `MaxPro`, `DiceDesign`, `SLHD` | ◐ random LHS, Sobol only | GPU-scale distance search | W13 |
| Orthogonal arrays, Taguchi arrays, covering arrays | Taguchi Arrays, Covering Arrays | DoE.base `oa.design` | ❌ | — | W15 |
| Nonlinear design (local, Bayesian) | Nonlinear Design | ICAOD, acebayes | ◐ local via `jacobian_features` | autograd; NN models; GPU prior MC | W17–W18 |
| Minimax / robust designs | — | ICAOD | ❌ | certificate; links to robust optimisation | W19 |
| Approximate designs and rounding | — | OptimalDesign (REX, KL) | ❌ | GPU weights + equivalence certificate | W17 |
| Discrete choice / MaxDiff designs | Choice Design, MaxDiff | `idefix` | ❌ | — | W22 |
| Incomplete block (BIBD, alpha), crossover designs | — | agricolae, crossdes | ❌ | — | backlog (Scientist decides after W16) |
| Sequential / closed-loop design with a model | — | — | ✅ `next_batch` (linearized) | autograd + BoTorch adapter | done; adapter W21 |
| Differentiable continuous design (no candidate grid) | — | — | ❌ | unique to TorchOED | W21–W22 |
| Measurement systems analysis (MSA) designs | MSA Design | — | ❌ | — | backlog |

Weeks 12, 16, 20 and 24 are consolidation weeks (refresh this matrix, published-example matches, release).

## Known documentation debt (week 12 consolidation)
`website/api.md` lists `optimal_design` without `blocks`, `whole_plots`, `hard`, `eta`; `docs/ALGORITHM_COVERAGE.md` still marks `optimal_design` PLANNED and definitive screening DEFERRED; older milestone docs (`docs/M1_API.md` … `docs/M3C_API.md`) should be merged or archived.
