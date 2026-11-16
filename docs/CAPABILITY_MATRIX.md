# Capability matrix (Completeness rule, VISION.md; baseline `main@4841b68`, 2026-11-16)

Every feature PR updates its row. Cell legend: ✅ done with evidence · ◐ partial (say what is missing) · ❌ missing · N/A not applicable (reason) · NR external reference NOT_RUN. Columns follow the Completeness rule: **Crit** D/A/I · **Aug** `fixed_ids` · **Con** candidate constraints / categorical coding · **Cert** efficiency certificate · **Diag** `design_diagnostics` incl. power · **Dev** CPU float64 / measured GPU · **Ext** external published reference · **Docs** API + guide snippet · **HO** held-out case.

## P1 / P1.5 — engine and model-driven design
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| `optimal_design` (plain exact) | ✅ D A I | ✅ | ✅ | ✅ KW dual floor | ✅ | ✅ CPU; GPU measured for D only (`w5-crossover`) | ✅ PyDOE, AlgDesign | ✅ | ✅ H1–H8 | done; A/I GPU timing → W13 |
| `jacobian_features` + `next_batch` | ✅ D A I | ✅ | ✅ | ✅ linearized only | ✅ | ◐ CPU only measured | ❌ no published nonlinear design reproduced | ✅ | ✅ H9–H10 | W13 (published nonlinear examples) |
| criterion functions | ◐ D A E c evaluated; G, V, I-as-function missing | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | W11 |

## P2 — industrial DOE (completion weeks 9–11)
| Capability | Crit | Aug | Con | Cert | Diag | Dev | Ext | Docs | HO | Week |
|---|---|---|---|---|---|---|---|---|---|---|
| blocking `blocks=` | ✅ D A I (#228) | ❌ | ✅ | ◐ local certificate only, no efficiency floor | ❌ diagnostics ignore blocks | ◐ CPU float64 only | ✅ AlgDesign `optBlock` (H11) | ✅ | ✅ H11 | **W9** |
| split-plot `whole_plots=` | ◐ D only | ❌ | ✅ | ◐ local certificate only | ❌ no GLS SPV / power | ◐ CPU float64 only | ✅ skpr `gen_design` (`w8-skpr`) | ✅ | ✅ H14 | **W9** |
| blocks + split-plot together | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | W9 if the gate covers it, else W11 |
| definitive screening | N/A constructor | N/A | N/A | N/A (exact orthogonality proved) | ✅ | ✅ | ◐ Paley orders only (q prime); other k raise | ✅ | ✅ H12 | W10 (non-Paley conference matrices) |
| fractional factorial by generators / resolution, fold-over, alias matrix | ❌ | ❌ | ❌ | N/A | ❌ alias diagnostics | ❌ | ❌ FrF2 | ❌ | ❌ | **W10** |
| Plackett–Burman, central composite (incl. blocked), Box–Behnken | ❌ | N/A | N/A | N/A | ❌ | ❌ | ❌ rsm / published tables | ❌ | ❌ | **W10** |
| mixtures: simplex lattice | N/A constructor | N/A | N/A | N/A | ✅ | ✅ | ❌ | ✅ | ✅ H15 | done |
| mixtures: simplex centroid, constrained (bounds, extreme vertices), optimal mixture D/A/I with Scheffé models | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ JMP/mixexp examples | ◐ manual Scheffé column selection documented | ❌ | **W11** |
| mixture-process variables | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **W11** |
| covariates (fixed per-run values) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **W11** |
| alias-aware and compound (e.g. D+I) criteria | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ JMP alias-optimal | ❌ | ❌ | **W11** |
| power | ◐ per-coefficient t-test, OLS, σ = 1 | N/A | N/A | N/A | ◐ no blocked/split-plot (GLS) power, no effect-group F-test | ✅ | ✅ SciPy (H16) | ✅ | ✅ H16 | **W9** (structured), W11 (F-test) |
| space filling: random LHS, Sobol | N/A | N/A | N/A | N/A | N/A | ✅ | ✅ | ✅ | N/A | done; maximin/MaxPro LHS → backlog |

## P3 — torch-native frontier (weeks 13–15)
| Capability | Status | Week |
|---|---|---|
| approximate designs (multiplicative / Frank–Wolfe weights) on GPU + equivalence-theorem certificates + efficient rounding | ❌ | W13 |
| nonlinear local designs reproduced from the literature (ICAOD examples) | ❌ | W13 |
| Bayesian / pseudo-Bayesian designs by batched prior Monte Carlo | ❌ | W14 |
| minimax / maximin-efficiency robust designs | ❌ | W15 |
| GPU-scale exchange to 10⁶ candidates | ◐ 10⁵ measured for D | W15 |

## P4 — ecosystem and 1.0 (weeks 17+)
| Capability | Status | Week |
|---|---|---|
| BoTorch/Ax adapter for sequential design | ❌ | W17 |
| gradient-based continuous design | ❌ | W17–18 |
| API freeze, semver, deprecation policy, 1.0 on PyPI | ❌ (TestPyPI 0.6.0; 0.7.0 in progress) | W19–20 |
| JOSS paper | ❌ | W20 |

## Known documentation debt (week 12 consolidation)
`website/api.md` lists `optimal_design` without `blocks`, `whole_plots`, `hard`, `eta`; `docs/ALGORITHM_COVERAGE.md` still marks `optimal_design` PLANNED and definitive screening DEFERRED; older milestone docs (`docs/M1_API.md` … `docs/M3C_API.md`) should be merged or archived.
