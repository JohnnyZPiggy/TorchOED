# Report index

One row per report directory. `current` retains distinct acceptance evidence; it does not claim historical timings describe today's source. Benchmark rows marked superseded remain the record for their original SHA/runtime. No evidence is moved or deleted.

| Directory | Measurement / producing PR | Status |
|---|---|---|
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
