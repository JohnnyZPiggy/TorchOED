# Week 12 goal — 2026-11-19 → 2026-11-26 (CONSOLIDATION: P2 exit — published-example matches, matrix and documentation refresh, 0.8.0 TestPyPI)

**Auto-advance (Architect, 2026-11-19).** Week 11 goals D1–D3 were met on 2026-11-19:
- D1 mixtures: 18/18 cells (#281, #284, #306, #350).
- D2 blocks with whole plots and covariates: 18/18 (#297, #310, #346).
- D3 compound and alias criteria: 9/9, with criterion-functions Crit ✅ and power Diag ✅ (#290, #302, #314, #322, #351).

Held-out H24–H29 were recorded as-is (#329, #334, #336):
- H24 PASS (Hausdorff 7.15e-8).
- H25 AlgDesign D MATCH.
- H26 skpr NOT_RUN.
- H27–H29 MATCH exhaustive, Δ = 0.

The corrected skpr Alias call (#351) MATCHes the exact front (768, 1/3). Baseline: `main` at the START of this goal's commit task.

**Why this week:** week 12 is a consolidation week (VISION.md weekly cadence), so it adds no new features. The P2 exit evidence in VISION.md asks for every P2 matrix row complete (or Scientist-deferred) and **≥ 8 published JMP/skpr/AlgDesign/FrF2/rsm examples** matched or beaten on one shared metric. Today only one example qualifies: the skpr `gen_design` Rd example in `reports/w8-skpr`. The other external comparisons ran the reference tools on TorchOED-chosen inputs. The parity checklist, `website/api.md`, `docs/ALGORITHM_COVERAGE.md` and `docs/RELEASE_READINESS.md` also lag the merged code. The last release was 0.7.0 (week 8).

A **published example** means a call printed in a reference package's help page (Rd `\examples`) or vignette at the recorded version, or in a cited book or paper (author, title, edition, page). Its inputs are copied from that source into a repo file, never transcribed from memory. It is run with recorded versions and re-scored by one shared TorchOED metric.

| Goal | Falsifiable target | Baseline |
|---|---|---|
| **E1 Published-example matches (P2 exit)** | One index `reports/w12-published/summary.md` that lists every published example with: source and version or edition; frozen input file; the reference call verbatim; the shared metric; and TorchOED's value and verdict (MATCH, TorchOED better, NO MATCH, or NR with the verbatim error). Prior reports that ran a documented example count only if the source line is cited. New examples run in one batched pinned-controller Colab R session. They cover at least 4 of AlgDesign (`optFederov`, `optBlock`), FrF2 (`FrF2`, `pb`), rsm (`ccd`, `bbd`), mixexp (`Xvert`, `SLD`, `SCD`), daewr (`DefScreen`) and skpr. Measured number: published examples MATCH or TorchOED better, target **≥ 8**. NO MATCH and NR are reported as-is and do not count | 1 (w8-skpr) |
| **E2 Matrix, parity checklist and documentation refresh** | (a) Every parity-checklist row whose status lags the P2 matrix rows is corrected, with evidence links. (b) The documentation debt listed in `docs/CAPABILITY_MATRIX.md` is closed: `website/api.md` lists every exported public name and keyword merged through week 11; `docs/ALGORITHM_COVERAGE.md` no longer marks merged capabilities PLANNED; the milestone API docs `docs/M1_API.md` … `docs/M3C_API.md` are merged into one historical file with every link updated; `docs/RELEASE_READINESS.md` reflects 0.8.0. (c) Each remaining ◐ cell of a P2 row is either completed with evidence (blocking and split-plot **Dev**: one measured Colab T4 GPU D/A/I search result against the same-host CPU, quality equal and time recorded as-is) or listed in one table for Scientist deferral, with its reason and the week that would close it. Measured number: stale parity rows plus open doc-debt items, target **0**; open P2 ◐ cells not completed and not listed, target **0** | stale parity rows counted by the E2 task (at least Evaluate Design, alias/compound, mixture and covariates); 4 doc-debt items; 4 P2 ◐ cells (blocking Dev, split-plot Dev, DSD Ext, mixture Con) |
| **E3 Release 0.8.0 on TestPyPI** | First, one fully green CI run on `main`, including `test (macos-latest, …)`, recorded by run id. Then: version 0.8.0; the CHANGELOG `Unreleased` section becomes `[0.8.0]`; the `testpypi.yml` workflow publishes; a clean-venv install from TestPyPI runs `examples/cpu_quickstart.py` and the industrial guide snippets, and the outputs are recorded. The Engineer pushes the tag in an Engineer task. Test-suite wall time on `ubuntu-latest, 3.13` is recorded before and after any test cleanup; any test change keeps every behaviour test and must not slow the suite. Measured number: release steps done, target **4/4** (green main incl. macOS, version + CHANGELOG, TestPyPI publish, clean install smoke) | 0/4 (0.7.0 on TestPyPI) |

**Held-out additions.** These are frozen here, before any week-12 task merges. They are never used for tests, tuning or dispatch choices. They run once at week end and are reported as-is.
- **H30 (A-optimal special-cubic mixture):** the TorchOED A-optimal 8-run design for the q = 3 special-cubic Scheffé model (`x₁, x₂, x₃, x₁x₂, x₁x₃, x₂x₃, x₁x₂x₃`) on the 7 simplex-centroid candidates, with default restarts.
  - Shared metric: `tr((FᵀF)⁻¹)` and the certificate floor.
  - Reference: exhaustive exact `Fraction` enumeration over all C(14, 8) = 3003 multisets of size 8 (dev-only, in the report).
- **H31 (Plackett–Burman 28):** TorchOED `plackett_burman` with 28 runs and 27 factors.
  - Shared metric: `XᵀX = 28·I` exactly, and the multiset of `|J₃|` values over all 2,925 column triples.
  - Reference: R FrF2 `pb(nruns = 28, randomize = FALSE)` with `sessionInfo()`, re-scored by the same function.
- **H32 (three-block D-optimal quadratic):** the TorchOED D-optimal 12-run design with `blocks` of sizes (4, 4, 4) for the full quadratic model in two factors on the 3² grid at levels −1, 0, 1, with default restarts.
  - Shared metric: the blocked information `det(M_B)^(1/6)` with fixed block effects, and the dual efficiency floor, computed by one function for both designs.
  - Reference: R AlgDesign `optBlock(~ quad(x1, x2), withinData = <the 9 candidates frozen as a CSV>, blocksizes = c(4, 4, 4), criterion = "D")` with `sessionInfo()`, re-scored by the same function.

If a reference call fails or its versions are not recorded, its column is `NOT_RUN`, with the error verbatim.

**Timing rule (durable, Scientist 2026-11-07):** per restart unless a goal says otherwise.

**Safeguards:**
- No new public functions or keywords. Refactors keep every behaviour test and every D/A/I result unchanged. Any change to search code reruns `tools/bench_b0.py`, with no time regression and identical D results.
- Published examples are cited by source line or page and frozen as repo data, never transcribed from memory. No claim of JMP parity beyond reproduced references.
- External references follow the VISION.md oracle rules. Only one Colab task is in flight at a time.
- No tuning on held-out cases. Historical reports and evidence are preserved, never deleted.
- Releases and tag pushes are Engineer tasks. Real PyPI waits for the Scientist.

**Out of scope this week:**
- New capabilities: space-filling designs (week 13), Bayesian "If Possible" terms (week 14), split-split-plot, strip-plot, minimum-aberration and blocked fractional factorials, orthogonal arrays and GLM power (week 15).
- Non-box mixture regions; new DSD sizes.
- Analysis and Studio (P2.5), P3.
- Real PyPI and Issue #13.
