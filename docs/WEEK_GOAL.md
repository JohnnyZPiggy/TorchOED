# Week 4 goal — 2026-11-07 → 2026-11-14 (consolidation week + P1 A/I speed carry-over)

**Scientist decision (2026-11-07, in chat with the Architect):** W1 timing is compared **per restart** (TorchOED `restarts=1` time vs one PyDOE run), quality unchanged. Under that definition week 3 is **complete**: W1 8/8 (the three remaining cases are 1.1×, 1.3× and 0.38× per restart), W2 median D floor 0.932 (#115), W3 held-out H1–H8 quality 17/17 (#122, `reports/` week-3 held-out), W4 0.6.0 on TestPyPI (#119). Baseline now: `main@39b8c56`, v0.6.0.

**Why this week:** it is the 4th week, so a consolidation week (no new public features). The Scientist also authorized one P1 carry-over: on the held-out set, every A/I row passed quality but failed time by 48×–1420× (H8 A/I ≈ 6.6 ks per call), because the A/I two-swap phase scans the whole N₂ neighbourhood exactly with no pruning bound. Fixing that is performance work on an existing feature, not a new feature.

Every Engineer dispatch must name the goal (C1–C4) it serves. Dev numbers come from `tools/bench_b0.py` only; held-out cases are never used for tuning.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **C1 A/I speed (P1 carry-over)** | Gate first: an A/I two-swap pruning or screening rule (analogue of `docs/M3B2_K2_SEARCH_GATE.md` §1.3 for the L-criteria of `docs/W1_GENERIC_CRITERIA_GATE.md`), PROVED and passing the independent review; if no proof is found, a `[Mentor]` Issue instead and C1 is reported not met. Then: A and I median time ≤ 2× D on every `bench_b0` case where both run (the W1 gate's own target), per restart; final A/I designs and objectives identical to `main@39b8c56`; exhaustive `Fraction` oracles unchanged | A/I ≫ D (first task measures the exact per-case A/I : D ratios on `bench_b0`) |
| **C2 Benchmark report** | `tools/bench_b0.py` reports a per-restart time column next to the existing total; `reports/` gets a one-page `README.md` index (one line per report directory: what, which PR, still current or superseded) | totals only; 26 report directories, no index |
| **C3 Code and test consolidation** | No public API change. Delete dead private code (each deletion justified by `git grep` showing no caller outside tests of that code); full CPU test suite wall time on CI ubuntu py3.13 reduced ≥ 25% without removing any required oracle test; `docs/` stays ≤ 40 files and AGENTS.md ≤ 80 lines | measure CI time on `39b8c56` |
| **C4 Local housekeeping** | The Engineer deletes its own task-owned local residue listed in Issue #83 (`.sae-*` venvs and control directories, untracked `reports/w95-head/`, `reports/w114-pair/`) and from now on deletes task-owned local directories when its PR merges | ≈ 14 leftover paths on the Engineer workspace |

**C1 time-box (Scientist decision 2026-11-09):** C1 work stops at the end of this week (2026-11-14). Whatever C1 reaches by then is the final P1 A/I result and is reported as-is (met or not met). C1 does **not** carry over as a week-5 goal: the week-5 draft starts VISION.md P1.5 (model-driven closed loop + scale crossover experiment). Further A/I speed work may continue only as background, at most one S task per week, never ahead of the week's goals.

**Timing rule (durable, Scientist decision 2026-11-07):** all time comparisons against PyDOE or R are per restart unless a goal says otherwise; total time for TorchOED's default restarts may be reported but is not a pass/fail criterion.

**Safeguards this week:** any gate change passes an independent fresh-context review before merge; C1 changes must keep D results byte-identical and re-run `tools/bench_b0.py` with no D time regression; held-out cases are not run this week.

**Out of scope this week:** new public functions or parameters, P2 features (split-plot, blocking, mixture, …), GPU work, Julia, real PyPI, Issue #13.
