# Report index

One row per report directory. `current` retains distinct acceptance evidence; it does not claim historical timings describe today's source. Benchmark rows marked superseded remain the record for their original SHA/runtime. No evidence is moved or deleted.

| Directory | Measurement / producing PR | Status |
|---|---|---|
| [w15-frf2-ext](w15-frf2-ext/) | W15 C2 frozen FrF2 2.3-5 references (#462) | current: 6 recorded, BL2 NOT_RUN verbatim; no search verdict |
| [w12-heldout](w12-heldout/) | current (W12 held-out H30–H32; no tuning) | H30 MATCH Fraction exhaustive; H31/H32 R NOT_RUN (workflow version assertion); 3 local oracles PASS; #368 BLOCKED |
| [w12-published](w12-published/) | W12 published-example source index and frozen inputs (#353) | 8/13 published MATCH; 1 NO MATCH; 4 NR (missing packages); two TorchOED oracles PASS; one Colab session cleaned up (#361) |
| [w11-heldout2](w11-heldout2/) | current (W11 frozen H27–H29; #336) | mixture-process D, covariates A and alias Ψ MATCH exhaustive references; Δ = 0, counts and dense re-scores 6 PASS; no tuning |
| [w11-heldout](w11-heldout/) | current (W11 held-out H24–H26; no tuning) | H24 frozen call 6 vertices, unfiltered Fraction PASS, Hausdorff 7.15e-8; H25 AlgDesign D MATCH, reference floor 0.9139763584432251 (two evaluation checks PASS, #334); H26 skpr NOT_RUN (errors); five original oracles PASS, cleanup confirmed |
| [b0](b0/) | Initial fixed CPU D comparison against PyDOE 1.6.0; [#42](https://github.com/JohnnyZPiggy/TorchOED/pull/42) | superseded by b1 |
| [b1](b1/) | G-based score-first / verify-top exchange; [#46](https://github.com/JohnnyZPiggy/TorchOED/pull/46) | superseded by b2 |
| [b2](b2/) | D-efficiency lower bounds and CPU comparison; [#53](https://github.com/JohnnyZPiggy/TorchOED/pull/53) | superseded by b3 |
| [b3](b3/) | Guarded pair-bound pruning; [#56](https://github.com/JohnnyZPiggy/TorchOED/pull/56) | superseded by b5 |
| [b4](b4/) | random2000 CPU G2 scale measurement; [#60](https://github.com/JohnnyZPiggy/TorchOED/pull/60) | current (distinct scale protocol) |
| [b5](b5/) | Saturated two-swap pruning / random500 speed; [#64](https://github.com/JohnnyZPiggy/TorchOED/pull/64) | superseded by b6 |
| [b6](b6/) | Closed-form pruned survivor scores; [#67](https://github.com/JohnnyZPiggy/TorchOED/pull/67) | superseded by b8 |
| [b8](b8/) | Survivor-chunk sizing and B8 CPU comparison; [#73](https://github.com/JohnnyZPiggy/TorchOED/pull/73) | superseded by w95-bench |
| [beta](beta/) | v0.5.0 beta capability / release evidence; [#70](https://github.com/JohnnyZPiggy/TorchOED/pull/70) | current (historical v0.5.0 record) |
| [m0](m0/) | Torch-only package spine and frozen upstream inventory audit; [#4](https://github.com/JohnnyZPiggy/TorchOED/pull/4) | current (acceptance record) |
| [m1](m1/) | DOE constructors and independent CPU oracles; [#6](https://github.com/JohnnyZPiggy/TorchOED/pull/6) | current (acceptance record) |
| [m2a](m2a/) | Information matrices and D/A/E criteria validation; [#8](https://github.com/JohnnyZPiggy/TorchOED/pull/8) | current (acceptance record) |
| [m2b](m2b/) | SPD c-optimality CPU / T4 validation; [#21](https://github.com/JohnnyZPiggy/TorchOED/pull/21) | current (acceptance record) |
| [m2b-multigpu](m2b-multigpu/) | c-optimality on T4/L4/A100/G4, not exchange-search support; [#26](https://github.com/JohnnyZPiggy/TorchOED/pull/26) | current (device evidence) |
| [m3a](m3a/) | Exact-ID one-swap CPU exchange oracles; [#10](https://github.com/JohnnyZPiggy/TorchOED/pull/10) | current (acceptance record) |
| [m3b1](m3b1/) | Simultaneous two-swap exact CPU score oracles; [#12](https://github.com/JohnnyZPiggy/TorchOED/pull/12) | current (acceptance record) |
| [m5a](m5a/) | CPU quickstart / API usability handoff; [#17](https://github.com/JohnnyZPiggy/TorchOED/pull/17), relocated in [#25](https://github.com/JohnnyZPiggy/TorchOED/pull/25) | current (historical handoff) |
| [s1-gpu](s1-gpu/) | S1 G2 CUDA exchange instrumentation and measured hardware results; [#77](https://github.com/JohnnyZPiggy/TorchOED/pull/77) | current (device-specific, no general GPU claim) |
| [w1-bench](w1-bench/) | Certified A/I one-swap baseline; [#92](https://github.com/JohnnyZPiggy/TorchOED/pull/92) | superseded by w105 |
| [w105](w105/) | A/I two-swap VND and unchanged D results; [#108](https://github.com/JohnnyZPiggy/TorchOED/pull/108) | superseded by w113 |
| [w113](w113/) | D/A/I dual-floor validation and timing; [#115](https://github.com/JohnnyZPiggy/TorchOED/pull/115) | current (bound evidence; later timings in w4-bench) |
| [w114](w114/) | Trusted D verification base/head timing pair; [#117](https://github.com/JohnnyZPiggy/TorchOED/pull/117) | superseded by w118 |
| [w118](w118/) | Lazy rows / combined dual-floor transfer pair, 40 identical D outputs; [#121](https://github.com/JohnnyZPiggy/TorchOED/pull/121) | superseded by w4-bench (timing convention) |
| [w131](w131/) | Exact pair bookkeeping base/head, profile and D/A/I equality; [#135](https://github.com/JohnnyZPiggy/TorchOED/pull/135) | baseline for w136 (original D timings retained) |
| [w136](w136/) | Closed-form screen timing, exact outputs and scoped H1/H2 witnesses; [#138](https://github.com/JohnnyZPiggy/TorchOED/pull/138) | current (development screen; C1 target unmet) |
| [w139](w139/) | Per-state screen tables, exact outputs, workspace fallback and N2 profile; [#142](https://github.com/JohnnyZPiggy/TorchOED/pull/142) | candidate (base screen retained; C1 unmet) |
| [w141](w141/) | Tangent-prefix kept/rescore fractions, exact outputs and full-tail H3/H4 margins; [#143](https://github.com/JohnnyZPiggy/TorchOED/pull/143) | candidate (C1 unmet; n=p timing tradeoff) |
| [w145](w145/) | Removal-tangent kept fractions, capacities and exact full-set H5/H4 margins; [#149](https://github.com/JohnnyZPiggy/TorchOED/pull/149) | candidate (batched; timing and margins pass) |
| [w3-heldout](w3-heldout/) | Frozen inputs and untuned H1-H8 final evidence in final/; [#102](https://github.com/JohnnyZPiggy/TorchOED/pull/102), [#112](https://github.com/JohnnyZPiggy/TorchOED/pull/112), [#122](https://github.com/JohnnyZPiggy/TorchOED/pull/122) | current (held-out record; no tuning) |
| [w3-probe](w3-probe/) | R AlgDesign reproduction probe; [#88](https://github.com/JohnnyZPiggy/TorchOED/pull/88) | superseded by w3-heldout |
| [w4-bench](w4-bench/) | Measured per-restart D/A/I ratios, retained 10-restart D rows; [#126](https://github.com/JohnnyZPiggy/TorchOED/pull/126) | current (development timing baseline) |
| [w95-bench](w95-bench/) | Fixed-run augmentation and unchanged D CPU pair; [#98](https://github.com/JohnnyZPiggy/TorchOED/pull/98) | superseded by w118 |
| [w147](w147/) | Square-design kept fractions, capacities and exact full-set H6/H4 margins; [#152](https://github.com/JohnnyZPiggy/TorchOED/pull/152) | candidate; timing and margins pass |
| [w151](w151/) | Seeded block thresholds, first-block theta and kept fractions; [#154](https://github.com/JohnnyZPiggy/TorchOED/pull/154) | negative (not merged; measured source `009950e` on branch `perf/w4-seed-theta-151`) |
| [w153](w153/) | Rule-e prefix release, capacities and timing comparison; [#156](https://github.com/JohnnyZPiggy/TorchOED/pull/156) | candidate; exact outputs and timing pass |
| [w155](w155/) | Non-hoisted row storage, capacities and timing comparison; [#158](https://github.com/JohnnyZPiggy/TorchOED/pull/158) | candidate; exact outputs and timing pass |
| [w157](w157/) | Top-8 probe-first screened tiles, final-mask/probe counts and timing comparison; [#160](https://github.com/JohnnyZPiggy/TorchOED/pull/160) | negative (not merged; measured source `aeae4e8` on branch `perf/w4-probe-first-157`) |
| [w159](w159/) | Base A/I screened/kept/gathered counts and per-kernel cProfile, seed0 on four slow cases; Refs #159 | candidate (evidence only; source `0213928`) |
| [w162](w162/) | Empty screened-tile skip, exact outputs and gather calls; [#164](https://github.com/JohnnyZPiggy/TorchOED/pull/164) | candidate (paired D timing8/8 pass; measured head `33de5a9`) |
| [w163](w163/) | A/I:D at1M/4M/16M workspace, exact outputs, gather counts and peak RSS; Refs #163 | candidate (evidence only; default unchanged) |
| [w165](w165/) | Six random500 D/A/I non-N₂ profiles, disjoint accounting, calls and floor iterations; Refs #165 | candidate (evidence only; source `6b4854f`) |
| [w171](w171/) | A/I exponent-1 dual floor and 1e-3 stop; [#172](https://github.com/JohnnyZPiggy/TorchOED/pull/172) | candidate; D timing8/8, exact outputs, C1 5/8 |
| [w168](w168/) | Final W4 C1 paired benchmark against `39b8c56`; Refs #168 | current (final W4 C1 result); 5/8, exact outputs pass |
| [w174](w174/) | Nine remaining-gap D/A/I profiles, exact outputs, disjoint accounting and dominant N₂; Refs #174 | candidate (evidence only) |
| [w4-heldout](w4-heldout/) | Final W4 H1-H8 TorchOED-only rerun, frozen references reused; Refs #176 | current (W4 held-out record; no tuning) |
| [w5-crossover](w5-crossover/) | W5 fixed CPU/GPU/PyDOE scale crossover, Refs #185 / #189 | run 1 inconclusive; [run 2](w5-crossover/run2-summary.md) recorded as-is |
| [w5-heldout](w5-heldout/) | Frozen H9 D/I and H10 D/A CPU record; Refs #187 | current (W5 held-out H9/H10; no tuning) |
| [w6-heldout](w6-heldout/) | Frozen H11 blocking, H12 screening and H13 diagnostics; Refs #199 | current (W6 held-out H11–H13; no tuning) |
| [w7-heldout](w7-heldout/) | Frozen H14 split-plot D, H15 mixture and H16 power; Refs #209 | current (W7 held-out H14–H16; no tuning) |
| [w9-heldout](w9-heldout/) | Frozen H19–H20 scored by shared dense metrics; Refs #239 | current (W9 held-out H19–H20; no tuning) |
| [w10-heldout](w10-heldout/) | Frozen H21–H23 and shared WLP/response-surface metrics; Refs #271 | current (W10 held-out H21–H23; no tuning) |
| [w11-covext](w11-covext/) | W11 covariates Ext (#339): TorchOED MATCH exhaustive 2048/25; skpr NOT_RUN (`undefined columns selected`), R 4.6.1 / skpr 1.9.2 recorded on authorized retry; both cleanups confirmed. | current (W11 covariates Ext; reference NOT_RUN) |
| [w11-aliasext](w11-aliasext/) | W11 alias Ext (#342): shared dense metric vs exact 1,716-multiset oracle | MATCH: skpr 1.9.2 and TorchOED both match exhaustive (det M, alias trace) = (768, 1/3); corrected aliaspower=2 (#349) |
| [w12-structgpu](w12-structgpu/) | #362 | BLOCKED: authorized replacement allocation TIMEOUT, cleanup confirmed; six local CPU cases PASS; Colab CPU/GPU NOT_RUN |
| [r-oracle-probe](r-oracle-probe/) | Manual GitHub Actions R oracle; R 4.5.2, rsm 2.10.6; [run 44500473486](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44500473486); Refs #372 | PASS: all 3 CCD/BBD CSVs byte-identical |
| [G0 Colab Pro probe](g0-probe/summary.md) | #385 | Retry2 PASS: numerical checks 7 PASS / 0 FAIL; cleanup PASS; previous failures preserved |
| [w13-published](w13-published/) | #392: four NR examples rerun on R 4.5.2 | Three MATCH, Xvert NO MATCH; published matches 8 -> 11 |
| [w13-spacefill-ext](w13-spacefill-ext/) | #404: R lhs 1.3.0 / MaxPro 4.1-2 / DiceDesign 1.10 | Diagnostics 16/16 MATCH; maximin better, MaxPro quality NO MATCH; Rd examples 3/3, published 11 → 14 |
| [w13-fff-ext](w13-fff-ext/) | #409: R support 0.1.7 on 511 frozen feasible points | Diagnostics + energy per metric (TorchOED better on φ₁₅, ψ, CD2; R on min distance, energy); Rd normal example not comparable; published 14 → 14 |
| [w13-sfgpu](w13-sfgpu/) | #422 | Same-host CPU / NVIDIA L4 space filling, 11 cells as-is; 22/22 validity PASS; cleanup confirmed |
| [w13-heldout](w13-heldout/) | #426 | Frozen H33–H35: H33 MATCH, H34 MATCH or TorchOED better, H35 MATCH; exhaustive integer/Fraction references and R MaxPro 4.1-2 |
| [w14-paper-ext](w14-paper-ext/) | #449: B1/B2 paper references | Both NR: full text not accessible verbatim; published count 16 unchanged |
| [w14-heldout](w14-heldout/) | #452: frozen H36–H39; #458: H39 R completion | H36–H38 exhaustive MATCH; H39 TorchOED better, MaxPro 23.66830 vs R 161.16186 (R 4.5.2 / support 0.1.7, run 44730389070); [completion](w14-heldout/h39-r/), frozen TorchOED unchanged |
| [w15-glmpower-ext](w15-glmpower-ext/) | #466: frozen skpr 1.9.2 GLM Monte Carlo power | G1–G3 recorded (skpr 1.9.2, R 4.5.2, run 44748014062) with MC SE and asymptotic Wald approximations (not verdicts); RD1 NOT_RUN (`should not be called with handlers on the stack`), Rd expression quoted from source only |
| [w15-skpr-ssp](w15-skpr-ssp/) | W15 nested split-plot skpr 1.9.2 | SS1 MATCH 1280/63, feasible with no repeated rows; SS2 published nested example MATCH shared GLS det 2072618.53036519, recorded variance ratios [1,1,1]. |
| [w15-glmpower-rd1](w15-glmpower-rd1/) | #486: published skpr eval_design_mc.Rd binomial example | RD1 NOT_RUN, child Rscript missing mbest; single R run 44796491427, R 4.5.2 / skpr 1.9.2; published count 18 unchanged |
| [w15-glmpower-rd1b](w15-glmpower-rd1b/) | #493: published skpr binomial RD1 retry with mbest 0.6.1 | NO MATCH: 2/3 coefficients within 4 combined SE; intercept z=8.516; R run 44814324521, R 4.5.2 / skpr 1.9.2; published count 18 unchanged |
| [w15-glmpower-rd2](w15-glmpower-rd2/) | #499: published skpr Poisson RD2 | NOT_RUN: Engineer dispatched before recipe commit; script absent at source SHA, run 44824241160; R 4.5.2 / skpr 1.9.2 / mbest 0.6.1; published count 18 unchanged |
| [w15-glmpower-rd2b](w15-glmpower-rd2b/) | #503: published skpr Poisson RD2 retry from committed recipe | MATCH: 3/3 coefficients within 4 combined SE; worst z=1.36656; R run 44831450343, R 4.5.2 / skpr 1.9.2 / mbest 0.6.1; published count 18 -> 19 |
| [g1-crossover](g1-crossover/) | #442: A100 G1 night 1, same-host D crossover and C3 CPU/CUDA timing | CUDA A/I out of contract, PyDOE errors recorded |
| [w15-heldout](w15-heldout/) | #512: frozen W15 H40–H42 | H40/H41 MATCH exact exhaustive GLS maxima; H42 MATCH FrF2 2.3-5 WLP, R 4.5.2, run 44868066239 |
| [w16-ai-scale](w16-ai-scale/) | #527: CPU A/I pair-scoring profile and bounded-workspace §2e fix, k = 5, n = 42, m ≤ 10⁴ | m = 5·10³ A/I finish (47.5 s / 162.3 s); m = 10⁴ A TIMEOUT, I 333 s; ≤ 120 s target NOT MET |
| [w10-dsd](w10-dsd/) | Corrected daewr DSD reference (#309) | k=6 MATCH; k=7 NO MATCH (shape); controller FAIL retained |
| [w10-rref](w10-rref/) | Week 10 FrF2/rsm reference freeze (#257, #261) | Catalogue 22/22 MATCH under §34.3, 7/22 original full-vector; PB 3/3 MATCH |
| [w11-mixai](w11-mixai/) | AlgDesign A/I mixture reference (#296) | A and I MATCH exhaustive; neither R design repeats candidates |
| [w11-mixlat](w11-mixlat/) | mixexp lattice/centroid archive reference (#321) | 7/7 MATCH; original CRAN attempt NOT_RUN retained |
| [w11-mixproc](w11-mixproc/) | Mixture-process AlgDesign reference (#313) | D and A MATCH; I TorchOED better, as-is |
| [w11-mixref](w11-mixref/) | Mixture archive retry (#289) | Xvert R1/R3 NO MATCH at 1e-12; D MATCH |
| [w11-skpr](w11-skpr/) | skpr nested-block probe (#293) | Reference NOT_RUN (both calls error); TorchOED re-scoring NOT_RUN |
| [w14-skpr-power](w14-skpr-power/) | skpr power across three run sizes (#437) | D8/D12/D16 NO MATCH at 1e-10; optional Rd NR |
| [w14-skpr-rd](w14-skpr-rd/) | Published skpr power examples (#443) | Simple and coffee MATCH at 1e-8; 10 comparisons PASS, 0 FAIL |
| [w8-heldout](w8-heldout/) | Frozen Week 8 H17–H18 (#221) | H17/H18 PASS against dense GLS oracles; no tuning |
| [w8-skpr](w8-skpr/) | Published skpr split-plot comparison (#217) | PASS: TorchOED = skpr, shared logdet delta 0 |
| [w9-skpr-i](w9-skpr-i/) | Published skpr I-optimal split-plot comparison (#237) | NOT_RUN; 0 comparisons; retry CLI_UNAVAILABLE |
| [w16-signature](w16-signature/) | Published AlgDesign D/A/I designs with dual efficiency floors (#541) | 13 checks PASS; D/I match, published A better; fixed D NOT_CERTIFIED |
