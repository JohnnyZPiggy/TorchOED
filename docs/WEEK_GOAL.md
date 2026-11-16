# Week 9 goal — 2026-11-17 → 2026-11-24 (VISION.md v2 phase P2 completion, part 1: structured designs for every criterion)

**Set by the Scientist (2026-11-16, in chat with the Architect: "全面点，仔细点").** VISION.md v2 adds the Completeness rule and [`docs/CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md); P2 is reopened until its matrix rows are complete, and P3 moves to weeks 13–15. Week 8 result is recorded in its Issue and `reports/`. Baseline: `main` at the START of this goal's commit task.

**Why this week:** blocking and split-plot, the two core industrial structures, exist only for the D criterion, without augmentation, without an efficiency floor and without structure-aware diagnostics. JMP's default for response-surface work is I-optimality, so D-only is not parity. This week completes the blocking and split-plot rows of the matrix.

Every Engineer dispatch names the goal (B1–B4) it serves and updates its matrix row in the same PR. No new root exports are needed (all goals extend `optimal_design` and `design_diagnostics`); any new keyword is decided in the gate.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **B1 Blocked A and I** | Gate first (independent review): A and I objectives on the block-adjusted information `M_B = Σ_b Σ_{i∈b}(fᵢ − f̄_b)(fᵢ − f̄_b)ᵀ` (I with the moment matrix of the candidate region, intercept absorbed), and the exchange update preserving block sizes. Then `optimal_design(..., criterion="A"/"I", blocks=…)`. Oracles: (a) objective equals the dense formula (1e-10); (b) exhaustive enumeration on a tiny case finds the same optimum value (Fraction); (c) D results of existing blocked tests unchanged | D only |
| **B2 Split-plot A and I** | Gate first (independent review): A and I on the GLS information `Fᵀ V⁻¹ F`, `V = I + η Z Zᵀ`, I with the candidate-region moment matrix. Then `optimal_design(..., criterion="A"/"I", whole_plots=…, hard=…, eta=…)`. Oracles: dense GLS objective (1e-10); tiny exhaustive enumeration; one whole-plot setting per plot; existing D results unchanged. External: one skpr 1.9.2 I-optimal split-plot call on Colab (re-scored by the shared GLS metric), or NR | D only |
| **B3 Augmentation and certificates for structured designs** | `fixed_ids` accepted with `blocks` and with `whole_plots` (fixed runs keep their block / whole plot as specified in the gate; returned IDs include them; free positions only are exchanged). Certificate: the gate either PROVES an efficiency lower bound for blocked and split-plot designs (each criterion), verified against exhaustive optima on ≥ 3 tiny cases, or the result carries `certificate="LOCAL_ONLY"` plus a documented reason and a [Mentor] Issue with the precise open question — never a silent local stop | no augmentation; local certificate only |
| **B4 Structure-aware diagnostics and power** | `design_diagnostics` accepts the same structure arguments and then uses `M_B` (blocks) or `Fᵀ V⁻¹ F` (split-plot) for SPV/FDS, VIF and per-coefficient power (degrees of freedom stated in the gate; for split-plot the whole-plot vs subplot denominators are named explicitly, or power is marked approximate with the rule used). Oracles: dense formulas (1e-10); unblocked results unchanged; one textbook split-plot power value reproduced or NR | ignores structure |

**Also this week (task 0, before B1):** commit VISION.md v2 and `docs/CAPABILITY_MATRIX.md` byte-identically with this goal.

**Held-out additions (frozen in their Issue before B1 or B2 code merges):** H19 a blocked I-optimal response-surface case (k = 3, full quadratic, 2 blocks of 8, 3-level grid) with R AlgDesign `optBlock` as reference if it supports the criterion (else NR); H20 a split-plot I-optimal case (1 hard-to-change, 2 easy factors, 4 whole plots × 3) with skpr 1.9.2 `gen_design(optimality="I")` as reference; both re-scored by one shared TorchOED metric and reported as-is.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:** every gate passes an independent fresh-context review before code; no claim of JMP parity beyond the published or reproduced references; external references follow the VISION.md oracle rules; no tuning on held-out cases; the matrix row of each goal is updated with evidence links in the same PR.

**Out of scope this week:** new constructors (week 10), mixtures / covariates / compound criteria (week 11), P3 (weeks 13–15), real PyPI, Issue #13.
