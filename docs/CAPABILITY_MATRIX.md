# Capability matrix (Completeness rule and parity checklist, VISION.md v3; baseline `main@4841b68`, 2026-11-16)

Every feature PR updates its row. Cell legend: ✅ done with evidence · ◐ partial (say what is missing) · ❌ missing · N/A not applicable (reason) · NR external reference NOT_RUN. Columns follow the Completeness rule: **Crit** D/A/I · **Aug** `fixed_ids` · **Con** candidate constraints / categorical coding · **Cert** efficiency certificate · **Diag** `design_diagnostics` incl. power · **Dev** CPU float64 / measured GPU · **Ext** external published reference · **Docs** API + guide snippet · **HO** held-out case.

## P1 / P1.5 — engine and model-driven design
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| `optimal_design` (plain exact) | ✅ D A I | ✅ | ✅ | ✅ KW dual floor; given-design evaluation `fixed_ids` length n (§80, #332) | ✅ | ✅ CPU; GPU measured for D only (`w5-crossover`; A100 k = 3/5/7, m ≤ 10⁵, `g1-crossover` #509) | ✅ PyDOE, AlgDesign | ✅ | ✅ H1–H8 | done; A/I GPU timing → W23 |
| `jacobian_features` + `next_batch` | ✅ D A I | ✅ | ✅ | ✅ linearized only | ✅ | ◐ CPU only measured | ❌ no published nonlinear design reproduced | ✅ | ✅ H9–H10 | W21 (published nonlinear examples) |
| criterion functions | ✅ D A E c G V I evaluated (§51, #290) | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | W11 |

## P2 — industrial DOE (completion weeks 9–11)
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| blocking `blocks=` | ✅ D A I (#228) | ✅ fixed_ids fill leading positions (#238) | ✅ | ✅ dual efficiency floor vs unblocked approximate optimum (#236) | ✅ M_B SPV/VIF/power (#234) | ◐ CPU float64 only | ✅ AlgDesign `optBlock` (H11) | ✅ | ✅ H11 | **W9** |
| split-plot `whole_plots=` | ✅ D A I (#232) | ✅ fixed_ids fill leading positions, plot label inherited (#238) | ✅ | ✅ dual efficiency floor vs OLS approximate optimum, conservative (#236) | ✅ GLS SPV/VIF; power exact (balanced orthogonal) else containment-df approx. (#234) | ◐ CPU float64 only | ✅ skpr `gen_design` (`w8-skpr`); I NR | ✅ | ✅ H14 | **W9** |
| split-split-plot and strip-plot | ✅ split-split-plot and strip-plot D A I (§102–§106, #459, #463) | ✅ fixed_ids, touched units retain labels (§120–§121, #488) | ✅ nested whole/subplot or crossed row/column labels, distinct ids (#459, #463) | ✅ OLS dual floor (§102.2, §105.2) | ✅ nested and crossed GLS diagnostics and containment power (§114–§118, #479, #483) | ✅ CPU float64, dense GLS + exhaustive optima (#459, #463) | ✅ skpr 1.9.2 SS1 MATCH 1280/63 and published gen_design.Rd nested example MATCH (#473, w15-skpr-ssp); strip-plot N/A (no skpr counterpart) | ✅ API (#459, #488), guides (#470, #475, #490, #494, #497) | ✅ H40 MATCH, H41 MATCH exhaustive Fraction optima (#514, [w15-heldout](../reports/w15-heldout/)) | W15 |
| blocks + split-plot together | ✅ D A I (§53, #297) | ✅ fixed_ids (§53.6, #297) | N/A search | ✅ dual floor (§53.7, #297) | ✅ M_C VIF/SPV/power, containment df (§74, #325) | ✅ CPU float64 | NR skpr nested-block calls error (#294) | ✅ API + guide (#297) | ✅ H26 recorded as-is, skpr NR (#329, [w11-heldout](../reports/w11-heldout/)) | **W11** |
| definitive screening | N/A constructor | N/A | N/A | N/A (exact orthogonality proved) | ✅ | ✅ | ◐ k 3–20, 23–30 (#248); k = 21, 22 impossible (no order-22 conference matrix); daewr 1.2.11 DefScreen k = 6 MATCH (signed column permutation), k = 7 NO MATCH (shape: daewr 15 × 7 vs. TorchOED 17 × 7) (#309, [w10-dsd](../reports/w10-dsd/)) | ✅ | ✅ H12 | W10 |
| fractional factorial by generators / resolution, fold-over, alias matrix | ✅ exact minimum-aberration WLP, feasible blocked fractions (§108–§109, #469) | N/A constructor | N/A fixed two-level design | N/A constructor | ✅ unstructured alias matrix ([#244](https://github.com/JohnnyZPiggy/TorchOED/issues/244)) | ✅ CPU float64, exhaustive WLP and block oracles (#469) | ✅ FrF2 catalogue 22/22; MA1–MA3, BL1, BL3, RD1 6/6 MATCH; BL2 both infeasible (#469, w15-frf2-ext) | ✅ [API](../website/api.md#candidate-construction), [guide](../website/industrial.md#fractional-factorials-and-fold-over) | ✅ H21 TorchOED (FrF2 reference NOT_RUN, [w10-heldout](../reports/w10-heldout/)); H42 minimum aberration MATCH FrF2 2.3-5 WLP (#514, [w15-heldout](../reports/w15-heldout/)) | **W10** |
| Plackett–Burman, central composite (incl. blocked), Box–Behnken | N/A constructors: PB (#255), CCD and BBD k = 3–7 (#258, #262) | N/A | N/A | N/A | ✅ CCD/BBD moment and CCD blocking oracles (#258); PB 12-run partial aliasing ±1/3 (#269) | ✅ CPU float64 | rsm 2.10.6: 7/7 non-held-out cases matched (#258, #262); FrF2 `pb` 3/3 exact, N = 12, 20, 24 (#261) | ✅ API + guide (#255, #258) | ✅ H22 blocked CCD, H23 BBD vs rsm ([w10-heldout](../reports/w10-heldout/)) | **W10** |
| mixtures: simplex lattice | N/A constructor | N/A | N/A | N/A | ✅ | ✅ | ✅ mixexp 1.2.7.1 SLD 4/4 MATCH (#321, [w11-mixlat](../reports/w11-mixlat/)) | ✅ | ✅ H15 | done |
| mixtures: simplex centroid, constrained (bounds, extreme vertices), optimal mixture D/A/I with Scheffé models | ✅ D A I via scheffe_features + optimal_design (§48, #284) | ✅ fixed_ids retained; replicated centroid exhaustive optimum (#284) | ◐ box bounds: vertices and face centroids (§45, #281); non-box regions out of scope | ✅ existing dual floor on Scheffé candidates (#284) | ✅ uncentred VIF, SPV, power (§77, #327) | ✅ CPU float64 | ✅ AlgDesign 1.2.1.2 optFederov D (#287), A and I (#296) MATCH on R1 ([w11-mixai](../reports/w11-mixai/)); mixexp 1.2.7.1 Xvert vertex counts agree, coordinates NO MATCH at 1e-12 (single-precision output) (#289, [w11-mixref](../reports/w11-mixref/)); mixexp 1.2.7.1 SCD 3/3 MATCH (#321, [w11-mixlat](../reports/w11-mixlat/)) | ✅ centroid, vertices, Scheffé optimal designs (#281, #284) | ✅ H24–H25 recorded as-is (#329, [w11-heldout](../reports/w11-heldout/)) | **W11** |
| mixture-process variables | ✅ D A I via mixture_process_features + optimal_design (§59, #306) | ✅ fixed_ids (§59.6, #306) | ✅ any mixture candidates (incl. mixture_vertices) × any process candidates (§59.2) | ✅ existing dual floor (§59.5, #306) | ✅ uncentred VIF, SPV, power (§77, #327) | ✅ CPU float64 | ✅ AlgDesign 1.2.1.2 optFederov on R1 × 2²: D and A MATCH, I TorchOED better (0.77483 vs 0.78095, lower average variance on the shared metric) (#313, [w11-mixproc](../reports/w11-mixproc/)) | ✅ API + guide (#306) | ✅ H27 D MATCH exhaustive, Δ = 0 (#336, [w11-heldout2](../reports/w11-heldout2/)) | **W11** |
| covariates (fixed per-run values) | ✅ D A I (§62, #310) | ✅ fixed_ids fill leading runs (§62.6) | ✅ any candidate F, no intercept | ✅ dual floor vs unstructured centered approximate optimum (§62.7) | ✅ M_Z correlation/VIF/SPV, t and F power, ν = n − 1 − c − p (§68, #318) | ✅ CPU float64 | NR skpr 1.9.2 `splitplotdesign` with singleton whole plots errors (`undefined columns selected`); TorchOED MATCH exhaustive 2048/25 (#344, [w11-covext](../reports/w11-covext/)) | ✅ API + guide (#310) | ✅ H28 A MATCH exhaustive, Δ = 0 (#336, [w11-heldout2](../reports/w11-heldout2/)) | **W11** |
| alias-aware and compound (e.g. D+I) criteria | ✅ compound D/A/I (§65, #314); alias-penalised D (§71, #322) | ✅ fixed_ids (§66, #314) | N/A search | ✅ compound dual floor (§65.5, #314); alias: D-efficiency floor only (§71.5) | ✅ alias matrix design_diagnostics(alias=) (§29, #244); alias_trace in results (§72, #322) | ✅ CPU float64 | ✅ skpr 1.9.2 `gen_design(optimality = "Alias", aliaspower = 2)` MATCH exhaustive (det M, ‖A‖²) = (768, 1/3); TorchOED MATCH; first call with nonexistent `aliasmodel` errored (#347, #351, [w11-aliasext](../reports/w11-aliasext/)) | ✅ compound + alias API and guide (#314, #322) | ✅ H29 alias Ψ MATCH exhaustive, Δ = 0 (#336, [w11-heldout2](../reports/w11-heldout2/)) | **W11** |
| power | ✅ per-coefficient t-test and effect-group F-test (§56, #302), OLS, σ = 1 | N/A | N/A | N/A | ✅ blocked + split-plot power (#234); effect-group F-test (§56, #302) | ✅ | ✅ SciPy (H16) | ✅ | ✅ H16 | **W9** (structured), W11 (F-test) |
| GLM power by simulation `glm_power(...)` | N/A evaluation | N/A | N/A | N/A | ✅ per-coefficient/group Wald power, MC SE, nonconverged/separated counts (§111–§112, #472) | ✅ CPU float64; A100 same host (g1-crossover, #509): nsim 10⁴ CPU 0.343 s / CUDA 0.370 s, nsim 10⁵ CPU 5.30 s / CUDA 0.422 s; CPU/CUDA powers within 4 combined MC SE | ✅ skpr G1–G3, 12/12 within 4 combined MC SE (#472, w15-glmpower-ext); published eval_design_mc.Rd binomial RD1b NO MATCH as-is (intercept z = 8.52 at skpr p = 1 with nsim 100; w15-glmpower-rd1b, #498); published Poisson RD2 NOT_RUN (dispatched before the recipe commit, run 44824241160; w15-glmpower-rd2, #502); published eval_design_mc.Rd Poisson RD2b MATCH 3/3 within 4 combined MC SE, worst z = 1.37 (w15-glmpower-rd2b, #506) | ✅ API (#472), guide (#482) | ❌ | W15 |
| space filling: random LHS, Sobol | N/A | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | done |
| space filling: maximin / MaxPro LHS | ✅ φ₁₅, ψ (#387) | N/A no fixed runs in an LHS | N/A box only; constrained → F3 | N/A no dual bound for LHS criteria | ✅ `space_filling_diagnostics` (#391) | ✅ CPU vs NVIDIA L4 same host, as-is (#422, [w13-sfgpu](../reports/w13-sfgpu/)) | ◐ R lhs 1.3.0 maximinLHS: TorchOED better; MaxPro 4.1-2 MaxProLHD: NO MATCH (#404, [w13-spacefill-ext](../reports/w13-spacefill-ext/)) | ✅ API (#387) + guide (#394) | ✅ H33 MATCH, H34 MATCH or TorchOED better (#426, [w13-heldout](../reports/w13-heldout/)) | W13 |
| space filling: uniform designs and diagnostics | ✅ CD2 (#391) | N/A no fixed runs in an LHS | N/A box only; constrained → F3 | N/A no dual bound for CD2 | ✅ `space_filling_diagnostics` (#391) | ✅ CPU vs NVIDIA L4 same host, as-is (#422, [w13-sfgpu](../reports/w13-sfgpu/)) | ✅ DiceDesign 1.10 mindist / phiP / C2 and MaxPro 4.1-2 MaxProMeasure agree ≤ 1e-12 (#404, [w13-spacefill-ext](../reports/w13-spacefill-ext/)) | ✅ API (#391) + guide (#398) | ✅ H35 MATCH (#426, [w13-heldout](../reports/w13-heldout/)) | W13 |
| space filling: fast flexible filling (constrained) | ✅ MaxPro point choice (§89.4) | N/A, no fixed runs (§91) | ✅ `A x ≤ b` + `disallowed` via `feasible_candidates` | N/A, no dual bound | ✅ `space_filling_diagnostics` | ✅ CPU vs NVIDIA L4 same host, as-is (#422, [w13-sfgpu](../reports/w13-sfgpu/)) | ✅ R support 0.1.7 sp on the frozen feasible sample, diagnostics + energy recorded as-is (#409, [w13-fff-ext](../reports/w13-fff-ext/)) | ✅ API (#395) + guide (#403) | ✅ H39 TorchOED better, MaxPro 23.668 vs R 161.162 (#456, #461, [w14-heldout](../reports/w14-heldout/h39-r/)) | W13 |
| "If Possible" terms: Bayesian D / I | ✅ D I (§93, #433) | ✅ real fixed_ids | ✅ distinct candidate ids | ✅ Bayesian dual floors (§93.4–§93.5) | N/A posterior diagnostics; ordinary design_diagnostics omits the prior, may be singular for n < p | ✅ CPU float64, Fraction exhaustive oracle (#433) | NR DuMouchel–Jones 1994 full text not accessible verbatim (#449, [w14-paper-ext](../reports/w14-paper-ext/)) | ✅ API (#433) + guide (#444) | ✅ H36 MATCH, H37 MATCH (#456, [w14-heldout](../reports/w14-heldout/)) | W14 |
| model-robust D over a set of models | ✅ weighted D (§96, #436) | ✅ fixed_ids | ✅ distinct ids, n ≥ union columns | ✅ robust + per-model dual floors (§96.5–96.6) | ✅ design_diagnostics per model on F[:, C_k] and returned ids | ✅ CPU float64, Fraction exhaustive oracles (#436) | NR Li–Nachtsheim 2000 full text not accessible verbatim (#449, [w14-paper-ext](../reports/w14-paper-ext/)) | ✅ API (#436) + guide (#448) | ✅ H38 MATCH (#456, [w14-heldout](../reports/w14-heldout/)) | W14 |

## P2.5 — analysis and TorchOED Studio (weeks 17–19, Scientist 2026-11-17)
| Capability | Status | Week |
|---|---|---|
| Mentor landscape review of statistics packages to reuse (CPU: statsmodels; GPU candidates named by the Scientist: PyStatistics, statgpu, Aurora-GLM) — existence, licence, maintenance, validation vs R | ✅ [Mentor §26](SCIENTIFIC_MENTOR_RESEARCH.md#26-p25-statistics-package-landscape-review-2026-11) (#517) | W16 (consolidation) |
| optional adapters to the chosen packages behind extras (never core dependencies) | ❌ | W17 |
| bootstrap intervals and batched fits for closed-loop rounds (GPU where measured faster) | ❌ | W18 |
| out of scope by decision: GP surrogates (BoTorch/Ax), general GLMM, survival analysis | N/A | — |
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
| Split-split-plot, strip-plot (very-hard-to-change) | Custom Design | skpr | ✅ split-split-plot and strip-plot D/A/I (#459, #463) | — | W15 |
| Evaluate Design: power, prediction variance, FDS, correlation map, efficiencies | Evaluate Design | skpr `eval_design`, `Vdgraph` | ✅ plain and structured SPV/VIF/power (#234, #318, #325); given-design evaluation (#332); G/V/I criteria (#290) | — | W9–W11 |
| Alias matrix, alias-optimal and compound criteria | Custom Design | FrF2 alias tools | ✅ alias matrix (#244), compound D/A/I (#314), alias-penalised D (#322); skpr MATCH (#351, [w11-aliasext](../reports/w11-aliasext/)) | — | W10–W11 |
| Power by simulation for GLMs (logistic, Poisson) | Sample Size Explorers | skpr `eval_design_mc` | ✅ `glm_power` CPU float64, per-coefficient/group Wald power; skpr 12/12 within 4 combined SE (#472); guide (#482); A100 timing (#509) | GPU Monte Carlo | W15 |
| Sample size explorer (power vs n) | Sample Size Explorers | `pwr`, skpr | ✅ `compare_designs` int entries, sequential shared-generator search; D/A/I/G, power and FDS (§99–§101, #441); exact oracle + seeded run-size tests (`test_compare_designs.py`); CPU float64; Ext skpr 1.9.2 [power](../reports/w14-skpr-power/) NO MATCH at 1e-10 as-is (R-side Δ), [Rd examples](../reports/w14-skpr-rd/) MATCH at 1e-8; API documented + guide (#455); unstructured only, no cross-design GPU batching | batched over n (open, §101) | W14 |
| Compare Designs (side-by-side efficiencies, power, FDS) | Compare Designs | skpr power | ✅ `compare_designs` relative total-information D/A/I/G, per-coefficient power, FDS (§99–§101, #441); Fraction-derived oracle (`test_compare_designs.py`); repeated IDs supported; CPU float64; Ext skpr 1.9.2 [power](../reports/w14-skpr-power/) NO MATCH at 1e-10 as-is (R-side Δ), [Rd examples](../reports/w14-skpr-rd/) MATCH at 1e-8; API documented + guide (#455); structured/Bayesian/model-robust comparisons excluded | — | W14 |
| "If Possible" terms, Bayesian D/I (DuMouchel–Jones) | Custom Design | — | ✅ D I (#433), guide (#444); Ext NR (#449) | certificate | W14 |
| Definitive screening | Definitive Screening | `daewr` DSD | ◐ k 3–20, 23–30 (#248); k = 21, 22 impossible; daewr k = 6 MATCH, k = 7 shape differs (#309, [w10-dsd](../reports/w10-dsd/)) | exact orthogonality proof | W10; Ext deferral requested |
| Fractional factorial by generators / resolution, minimum aberration, blocked FF, fold-over | Screening | FrF2, DoE.base | ✅ generators, fold-over, WLP, alias matrix; exact minimum aberration and blocked FF (#469); FrF2 6/6 feasible MATCH, BL2 both infeasible | — | W10; MA, blocked FF W15 (#469) |
| Plackett–Burman | Screening | FrF2 `pb` | ✅ N ≡ 0 mod 4, 4–48 (#255); FrF2 pb 3/3 exact (#261); partial aliasing (#269) | — | W10 |
| Group-orthogonal supersaturated designs | Screening | — | ❌ | — | backlog |
| Central composite (incl. blocked), Box–Behnken | Response Surface | `rsm` | ✅ CCD incl. blocks and BBD k = 3–7 (#258, #262); rsm 7/7 MATCH | — | W10 |
| Mixture: simplex lattice / centroid, extreme vertices, ABCD, constrained, mixture-process | Mixture Design | `mixexp` | ◐ lattice (#321), centroid and box vertices (#281), Scheffé D/A/I (#284), mixture-process (#306), uncentred diagnostics (#327); non-box / ABCD regions missing; [w11-mixai](../reports/w11-mixai/), [w11-mixproc](../reports/w11-mixproc/) | optimal mixture with certificate | W11; non-box deferral requested |
| Covariates (incl. hard-to-change covariates) | Custom Design | AlgDesign (fixed rows) | ◐ fixed per-run D/A/I (#310) and adjusted diagnostics/power (#318); hard-to-change covariates missing; skpr NR, exact MATCH (#344, [w11-covext](../reports/w11-covext/)) | — | W11; hard-to-change: Scientist decision pending |
| Space filling: LHS (maximin), MaxPro, uniform, sphere packing, fast flexible filling (constrained) | Space Filling Design | `lhs`, `MaxPro`, `DiceDesign`, `SLHD` | ◐ maximin / MaxPro LHS (#387), uniform CD2 + diagnostics (#391), constrained fast flexible filling (#395); R lhs maximin: TorchOED better, MaxProLHD NO MATCH; DiceDesign/MaxPro diagnostics MATCH (#404, [w13-spacefill-ext](../reports/w13-spacefill-ext/)); constrained R support as-is (#409, [w13-fff-ext](../reports/w13-fff-ext/)); CPU/L4 measured (#422), held-out H33–H35 MATCH or better (#426), H39 better (#461); sphere packing beyond maximin missing | GPU-scale distance search | W13 |
| Orthogonal arrays, Taguchi arrays, covering arrays | Taguchi Arrays, Covering Arrays | DoE.base `oa.design` | ❌ | — | Scientist decision pending |
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

## P2 exit: open cells

Listed for Scientist decision; listing is not a deferral approval. Open cells neither completed nor listed: 0 (4 listed).

| P2 row | Column | Missing | Why | Proposed close |
|---|---|---|---|---|
| blocking | Dev | Measured GPU D/A/I vs same-host CPU | CPU float64 only measured (#228); W12 T4 allocation timed out twice (#362) | One T4 retry after #368; else Scientist deferral |
| split-plot | Dev | Measured GPU D/A/I vs same-host CPU | CPU float64 only measured (#232); W12 T4 allocation timed out twice (#362) | One T4 retry after #368; else Scientist deferral |
| definitive screening | Ext | daewr k = 7 shape match | daewr 15 × 7 vs TorchOED 17 × 7 (#309, [w10-dsd](../reports/w10-dsd/)); k = 21, 22 impossible | Scientist deferral requested |
| constrained mixtures | Con | Non-box regions | Only box bounds, vertices and face centroids (#281) | Scientist deferral requested |

## P2+ open items for Scientist decision

Listed for Scientist decision; listing is not a deferral approval. Open cells/rows neither completed nor listed: 0.

| Item | Row / column | Status | Missing evidence | Why | Proposed close (week) |
|---|---|---|---|---|---|
| A/I GPU timing | P1 optimal_design / Dev | Partial despite ✅ | Same-host A/I timing | D only measured; CUDA A/I mixed-device error (#509, [g1-crossover](../reports/g1-crossover/)) | W23 GPU scale |
| Blocking GPU | P2 blocking / Dev | ◐ | Same-host D/A/I timing | CPU only; allocation failures (#362, [w12-structgpu](../reports/w12-structgpu/)) | W23 GPU scale or Scientist deferral requested |
| Split-plot GPU and I reference | P2 split-plot / Dev, Ext | ◐ / I NR within ✅ | GPU timing; skpr I comparison | CPU only; skpr I NOT_RUN ([w9-skpr-i](../reports/w9-skpr-i/)) | W23 GPU scale; Scientist deferral requested for I reference |
| Nested blocks reference | P2 blocks + split-plot / Ext, HO | NR; skpr NR within ✅ HO | Working external nested-block call / H26 reference | skpr errors (#294, [w11-skpr](../reports/w11-skpr/)); H26 recorded as-is (#329) | Scientist deferral requested |
| Fractional-factorial H21 reference | P2 fractional factorial / HO | NOT_RUN within ✅ | FrF2 H21 comparison | H21 TorchOED recorded as-is; H42 later MATCH (#514, [w15-heldout](../reports/w15-heldout/)) | Scientist deferral requested for H21 |
| DSD reference and factor coverage | P2 definitive screening / Ext; checklist DSD | ◐ | daewr k = 7 shape match; k = 21/22 coverage | 15×7 vs 17×7; order-22 conference matrix impossible (#309, [w10-dsd](../reports/w10-dsd/)) | Scientist deferral requested |
| Non-box mixture regions including ABCD | P2 mixtures / Con; checklist Mixture | ◐ | Non-box regions | Only box vertices/face centroids (#281); mixture Con remains open | Scientist deferral requested |
| Mixture vertex coordinates | P2 mixtures / Ext | NO MATCH within ✅ | Xvert coordinates at 1e-12 | mixexp single precision (#289, [w11-mixref](../reports/w11-mixref/)) | Scientist deferral requested |
| Covariate reference | P2 covariates / Ext | NR | External adjusted-design comparison | skpr singleton whole-plot error; exhaustive MATCH only (#344, [w11-covext](../reports/w11-covext/)) | Scientist deferral requested |
| Hard-to-change covariates | checklist Covariates | ◐ | Search/diagnostics with hard-to-change covariates | Fixed per-run covariates only (#310, #318) | Scientist deferral requested |
| GLM power held-out | P2 GLM power / HO | ❌ | Frozen H44 result | Filled by H44 at week end (#516); not run yet | W16 week end |
| GLM published reference discrepancies | P2 GLM power / Ext | NO MATCH / NOT_RUN within ✅ | RD1b agreement; valid RD2 dispatch | RD1b z = 8.52; RD2 dispatched before recipe commit (#498, #502); RD2b MATCH (#506) | Scientist deferral requested |
| Compare Designs / sample-size explorer limitations | checklist Compare Designs, Sample size explorer | NO MATCH within ✅; batching open | skpr power agreement; batched search over n | R-side power delta; sequential unstructured search only (#455, [w14-skpr-power](../reports/w14-skpr-power/)) | Scientist deferral requested |
| MaxPro external optimum | P2 maximin / MaxPro LHS / Ext; checklist Space filling | ◐ | MaxProLHD match | maximin better, MaxPro NO MATCH (#404, [w13-spacefill-ext](../reports/w13-spacefill-ext/)) | Scientist deferral requested |
| Sphere packing beyond maximin | checklist Space filling | ◐ | Sphere-packing design evidence | W13 implements LHS/CD2/constrained FFF only (#387, #391, #395) | Scientist deferral requested |
| Bayesian D/I paper | P2 If Possible / Ext | NR | Verbatim published example | DuMouchel–Jones full text inaccessible (#449, [w14-paper-ext](../reports/w14-paper-ext/)) | Scientist deferral requested |
| Model-robust paper | P2 model-robust D / Ext | NR | Verbatim published example | Li–Nachtsheim full text inaccessible (#449, [w14-paper-ext](../reports/w14-paper-ext/)) | Scientist deferral requested |
| Orthogonal / Taguchi / covering arrays | checklist Orthogonal arrays | ❌ | Constructors and external examples | No merged implementation | Scientist deferral requested |
| Group-orthogonal supersaturated designs | checklist Group-orthogonal | ❌ | Constructor and reference | No merged implementation | Scientist deferral requested |
| Incomplete block / crossover designs | checklist Incomplete block | ❌ | BIBD/alpha/crossover designs and references | Backlog; VISION places them in P4 | W25+ P4 or Scientist deferral requested |
| Measurement systems analysis designs | checklist MSA | ❌ | MSA design evidence | Backlog; no merged implementation | Scientist deferral requested |

P2+ signature on a published example: not shown

## Known documentation debt (week 12 consolidation)
Documentation debt is closed by #363 and #365.
