# Capability matrix (Completeness rule and parity checklist, VISION.md v3; baseline `main@4841b68`, 2026-11-16)

Every feature PR updates its row. Cell legend: ✅ done with evidence · ◐ partial (say what is missing) · ❌ missing · N/A not applicable (reason) · NR external reference NOT_RUN. Columns follow the Completeness rule: **Crit** D/A/I · **Aug** `fixed_ids` · **Con** candidate constraints / categorical coding · **Cert** efficiency certificate · **Diag** `design_diagnostics` incl. power · **Dev** CPU float64 / measured GPU · **Ext** external published reference · **Docs** API + guide snippet · **HO** held-out case.

## P1 / P1.5 — engine and model-driven design
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| `optimal_design` (plain exact) | ✅ D A I | ✅ | ✅ | ✅ KW dual floor | ✅ | ✅ CPU; GPU measured for D only (`w5-crossover`) | ✅ PyDOE, AlgDesign | ✅ | ✅ H1–H8 | done; A/I GPU timing → W23 |
| `jacobian_features` + `next_batch` | ✅ D A I | ✅ | ✅ | ✅ linearized only | ✅ | ◐ CPU only measured | ❌ no published nonlinear design reproduced | ✅ | ✅ H9–H10 | W21 (published nonlinear examples) |
| criterion functions | ✅ D A E c G V I evaluated (§51, #290) | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | W11 |

## P2 — industrial DOE (completion weeks 9–11)
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| blocking `blocks=` | ✅ D A I (#228) | ✅ fixed_ids fill leading positions (#238) | ✅ | ✅ dual efficiency floor vs unblocked approximate optimum (#236) | ✅ M_B SPV/VIF/power (#234) | ◐ CPU float64 only | ✅ AlgDesign `optBlock` (H11) | ✅ | ✅ H11 | **W9** |
| split-plot `whole_plots=` | ✅ D A I (#232) | ✅ fixed_ids fill leading positions, plot label inherited (#238) | ✅ | ✅ dual efficiency floor vs OLS approximate optimum, conservative (#236) | ✅ GLS SPV/VIF; power exact (balanced orthogonal) else containment-df approx. (#234) | ◐ CPU float64 only | ✅ skpr `gen_design` (`w8-skpr`); I NR | ✅ | ✅ H14 | **W9** |
| blocks + split-plot together | ✅ D A I (§53, #297) | ✅ fixed_ids (§53.6, #297) | N/A search | ✅ dual floor (§53.7, #297) | ❌ | ✅ CPU float64 | NR skpr nested-block calls error (#294) | ✅ API + guide (#297) | ❌ | **W11** |
| definitive screening | N/A constructor | N/A | N/A | N/A (exact orthogonality proved) | ✅ | ✅ | ◐ k 3–20, 23–30 (#248); k = 21, 22 impossible (no order-22 conference matrix); published DSD match NR: daewr DefScreen k = 6, 7 NOT_RUN, Colab ALLOCATION_FAILED ([#266](https://github.com/JohnnyZPiggy/TorchOED/issues/266)) | ✅ | ✅ H12 | W10 |
| fractional factorial by generators / resolution, fold-over, alias matrix | N/A constructor | N/A constructor | N/A fixed two-level design | N/A constructor | ✅ unstructured alias matrix ([#244](https://github.com/JohnnyZPiggy/TorchOED/issues/244)) | ✅ CPU float64 with exact WLP oracle ([P2 gate §28](P2_GATE.md#28-regular-fractional-factorials-fold-over-word-length-pattern-and-alias-matrix-w10-c1-part-1)) | ✅ FrF2 `catlg` first entries, 22/22 matched (#257, #251) | ✅ [API](../website/api.md#candidate-construction), [guide](../website/industrial.md#fractional-factorials-and-fold-over) | ✅ H21 TorchOED (FrF2 reference NOT_RUN, [w10-heldout](../reports/w10-heldout/)) | **W10** |
| Plackett–Burman, central composite (incl. blocked), Box–Behnken | N/A constructors: PB (#255), CCD and BBD k = 3–7 (#258, #262) | N/A | N/A | N/A | ✅ CCD/BBD moment and CCD blocking oracles (#258); PB 12-run partial aliasing ±1/3 (#269) | ✅ CPU float64 | rsm 2.10.6: 7/7 non-held-out cases matched (#258, #262); FrF2 `pb` 3/3 exact, N = 12, 20, 24 (#261) | ✅ API + guide (#255, #258) | ✅ H22 blocked CCD, H23 BBD vs rsm ([w10-heldout](../reports/w10-heldout/)) | **W10** |
| mixtures: simplex lattice | N/A constructor | N/A | N/A | N/A | ✅ | ✅ | ❌ | ✅ | ✅ H15 | done |
| mixtures: simplex centroid, constrained (bounds, extreme vertices), optimal mixture D/A/I with Scheffé models | ✅ D A I via scheffe_features + optimal_design (§48, #284) | ✅ fixed_ids retained; replicated centroid exhaustive optimum (#284) | ◐ box bounds: vertices and face centroids (§45, #281); non-box regions out of scope | ✅ existing dual floor on Scheffé candidates (#284) | ❌ | ✅ CPU float64 | ✅ AlgDesign 1.2.1.2 optFederov D (#287), A and I (#296) MATCH on R1 ([w11-mixai](../reports/w11-mixai/)); mixexp 1.2.7.1 Xvert vertex counts agree, coordinates NO MATCH at 1e-12 (single-precision output) (#289, [w11-mixref](../reports/w11-mixref/)) | ✅ centroid, vertices, Scheffé optimal designs (#281, #284) | ❌ | **W11** |
| mixture-process variables | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **W11** |
| covariates (fixed per-run values) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **W11** |
| alias-aware and compound (e.g. D+I) criteria | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ JMP alias-optimal | ❌ | ❌ | **W11** |
| power | ◐ per-coefficient t-test, OLS, σ = 1 | N/A | N/A | N/A | ◐ blocked + split-plot power ✅ (#234); no effect-group F-test | ✅ | ✅ SciPy (H16) | ✅ | ✅ H16 | **W9** (structured), W11 (F-test) |
| space filling: random LHS, Sobol | N/A | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | done; maximin/MaxPro LHS → backlog |

## P2.5 — analysis and TorchOED Studio (weeks 17–19, Scientist 2026-11-17)
| Capability | Status | Week |
|---|---|---|
| least-squares fit: coefficients, SEs, effect tests (t, F by term), lack of fit, residual diagnostics | ❌ | W17 |
| REML / mixed-model fit for blocked and split-plot designs (whole-plot vs subplot error) | ❌ | W17–W18 |
| prediction profiler data (prediction ± interval at a point, effect traces) | ❌ | W18 |
| desirability functions and multi-response optimisation | ❌ | W18 |
| Studio: factors/model → design → evaluate → compare → export | ❌ | W19 |
| Studio: import responses → fit → profiler; hosted demo | ❌ | W19 |

## P3 — torch-native frontier (weeks 21–23)
| Capability | Status | Week |
|---|---|---|
| approximate designs (multiplicative / Frank–Wolfe weights) on GPU + equivalence-theorem certificates + efficient rounding | ❌ | W21 |
| nonlinear local designs reproduced from the literature (ICAOD examples) | ❌ | W21 |
| Bayesian / pseudo-Bayesian designs by batched prior Monte Carlo | ❌ | W22 |
| minimax / maximin-efficiency robust designs | ❌ | W23 |
| GPU-scale exchange to 10⁶ candidates | ◐ 10⁵ measured for D | W23 |

## P4 — ecosystem and 1.0 (weeks 25+)
| Capability | Status | Week |
|---|---|---|
| BoTorch/Ax adapter for sequential design | ❌ | W25 |
| gradient-based continuous design | ❌ | W25–26 |
| API freeze, semver, deprecation policy, 1.0 on PyPI | ❌ (TestPyPI 0.7.0) | W27–28 |
| discrete choice designs (idefix-style) | ❌ | W26 |
| JOSS paper | ❌ | W28 |

## Parity checklist vs JMP 18 DOE and R (Scientist 2026-11-16)
One row per capability a practitioner would look for in JMP or the named R packages. "Advantage" names the TorchOED signature to demonstrate on that row where one applies.

| Reference capability | JMP | R reference | TorchOED row / status | Advantage to show | Week |
|---|---|---|---|---|---|
| Custom design, D/A/I, constraints, disallowed combinations, categorical | Custom Design | AlgDesign `optFederov` | ✅ `optimal_design` | certificate; GPU at 10⁵ | done |
| Augment design (add runs to existing) | Augment Design | AlgDesign `augment`-style via fixed rows | ✅ plain; ✅ blocks / whole plots (#238) | certificate on the augmented design | W9 |
| Blocking, fixed and random blocks | Custom Design | AlgDesign `optBlock`, skpr | ✅ D A I (#228); Cert ✅ (#236); Aug ✅ (#238) | certificate | W9 |
| Split-plot, hard-to-change factors | Custom Design | skpr `gen_design` | ✅ D A I (#232); Cert ✅ (#236); Aug ✅ (#238) | certificate; GPU | W9 |
| Split-split-plot, strip-plot (very-hard-to-change) | Custom Design | skpr | ❌ | — | W15 |
| Evaluate Design: power, prediction variance, FDS, correlation map, efficiencies | Evaluate Design | skpr `eval_design`, `Vdgraph` | ◐ unstructured only (W9-B4) | — | W9 |
| Alias matrix, alias-optimal and compound criteria | Custom Design | FrF2 alias tools | ◐ unstructured alias matrix ([#244](https://github.com/JohnnyZPiggy/TorchOED/issues/244)); alias-optimal and compound criteria open | — | W10–W11 |
| Power by simulation for GLMs (logistic, Poisson) | Sample Size Explorers | skpr `eval_design_mc` | ❌ | GPU Monte Carlo | W15 |
| Sample size explorer (power vs n) | Sample Size Explorers | `pwr`, skpr | ◐ power at fixed n | batched over n | W15 |
| Compare Designs (side-by-side efficiencies, power, FDS) | Compare Designs | — | ❌ | — | W15 |
| "If Possible" terms, Bayesian D/I (DuMouchel–Jones) | Custom Design | — | ❌ | certificate | W14 |
| Definitive screening | Definitive Screening | `daewr` DSD | ◐ k 3–20, 23–30 (#248); published DSD match open | exact orthogonality proof | W10 |
| Fractional factorial by generators / resolution, minimum aberration, blocked FF, fold-over | Screening | FrF2, DoE.base | ◐ generators, fold-over and WLP ([#244](https://github.com/JohnnyZPiggy/TorchOED/issues/244)); ✅ `(k, runs)` from FrF2 catalogue, 22/22 (#251) | — | W10 (MA, blocked FF → W15) |
| Plackett–Burman | Screening | FrF2 `pb` | ◐ N ≡ 0 mod 4, 4–48 (#255); FrF2 `pb` 3/3 matched (#261) | — | W10 |
| Group-orthogonal supersaturated designs | Screening | — | ❌ | — | backlog |
| Central composite (incl. blocked), Box–Behnken | Response Surface | `rsm` | ◐ CCD rotatable/orthogonal/faces/numeric α with cube + axial blocks; BBD k = 3–7 (#258, #262) | — | W10 |
| Mixture: simplex lattice / centroid, extreme vertices, ABCD, constrained, mixture-process | Mixture Design | `mixexp` | ◐ simplex lattice only | optimal mixture with certificate | W11 |
| Covariates (incl. hard-to-change covariates) | Custom Design | AlgDesign (fixed rows) | ❌ | — | W11 |
| Space filling: LHS (maximin), MaxPro, uniform, sphere packing, fast flexible filling (constrained) | Space Filling Design | `lhs`, `MaxPro`, `DiceDesign`, `SLHD` | ◐ random LHS, Sobol only | GPU-scale distance search | W13 |
| Orthogonal arrays, Taguchi arrays, covering arrays | Taguchi Arrays, Covering Arrays | DoE.base `oa.design` | ❌ | — | W15 |
| Nonlinear design (local, Bayesian) | Nonlinear Design | ICAOD, acebayes | ◐ local via `jacobian_features` | autograd; NN models; GPU prior MC | W21–W22 |
| Minimax / robust designs | — | ICAOD | ❌ | certificate; links to robust optimisation | W23 |
| Approximate designs and rounding | — | OptimalDesign (REX, KL) | ❌ | GPU weights + equivalence certificate | W21 |
| Discrete choice / MaxDiff designs | Choice Design, MaxDiff | `idefix` | ❌ | — | W26 |
| Incomplete block (BIBD, alpha), crossover designs | — | agricolae, crossdes | ❌ | — | backlog (Scientist decides after W16) |
| Fit Model: least squares, effect tests, lack of fit, residuals | Fit Model | `lm`, `rsm` | ❌ | batched fits; autograd | W17 |
| Mixed-model / REML analysis of split-plot and blocked experiments | Fit Model (REML) | `lme4`, `nlme` | ❌ | — | W17–W18 |
| Prediction profiler, desirability, multi-response optimisation | Prediction Profiler | `rsm`, `desirability` | ❌ | gradient-based optimisation of the fitted model | W18 |
| Interactive DOE workflow (design → evaluate → compare → analyse) | DOE platform GUI | — | ❌ TorchOED Studio | certificates and GPU shown in the UI | W19 |
| Sequential / closed-loop design with a model | — | — | ✅ `next_batch` (linearized) | autograd + BoTorch adapter | done; adapter W25 |
| Differentiable continuous design (no candidate grid) | — | — | ❌ | unique to TorchOED | W25–W26 |
| Measurement systems analysis (MSA) designs | MSA Design | — | ❌ | — | backlog |

Weeks 12, 16, 20, 24 and 28 are consolidation weeks (refresh this matrix, published-example matches, release).

## Known documentation debt (week 12 consolidation)
`website/api.md` lists `optimal_design` without `blocks`, `whole_plots`, `hard`, `eta`; `docs/ALGORITHM_COVERAGE.md` still marks `optimal_design` PLANNED; older milestone docs (`docs/M1_API.md` … `docs/M3C_API.md`) should be merged or archived.
