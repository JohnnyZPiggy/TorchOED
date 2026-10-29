# TorchOED — SAE V3.3 Experiment Protocol / Engineer Handoff

> **E0 2026-10-29 correction:** Codex delivered actual original-wheel data in [Draft PR #3](https://github.com/JohnnyZPiggy/TorchOED/pull/3). P1's original rank-12 mathematical hypothesis was disproved: the actual PyDOE design is rank **15/15**. See Scientific Mentor §25. The preserved initial E0 scope is historical; do not conflate a failed oracle prediction with a PyDOE bug.

> **Chief Architect | 2026-10-29 | EXPLORE → VALIDATE | Reuse-First | evidence over claims**
>
> References: [DEVELOPMENT_PLAN](DEVELOPMENT_PLAN.md), [SCIENTIFIC_MENTOR_RESEARCH](SCIENTIFIC_MENTOR_RESEARCH.md) §§17–24, [ARCHITECT_ECOSYSTEM_SURVEY](ARCHITECT_ECOSYSTEM_SURVEY.md), [M0 Issue #1](https://github.com/JohnnyZPiggy/TorchOED/issues/1).
>
> **Scope boundary:** An experiment specification for an already active Codex Engineer; **not** authorization to edit M0 package sources, auto-merge, open Actions/paid services or deploy. The user reported the SAE cold/warm protocol passed. At documentation time the remote M0 Issue #1 was OPEN, main HEAD `82a2fae8809e78d1f02d8e44cbcd0ac22992890b`, zero PRs, and no package implementation merged; no GitHub-side evidence of a Codex CLAIM/PR was visible. Recheck before Engineer CLAIM.

## 1. Decision and staged work packages

| ID | Objective | Status at design | Dependency | Engineer deliverables |
|---|---|---|---|---|
| **E0** | **Real PyDOE 1.6.0 wheel parity** against independent numerical/structural math oracles | **READY TO CLAIM** | Standalone experiment branch; no dependency on completed M0 | Isolated CPU comparator, provenance+SHA record, deterministic JSON/MD results, tiny tests, draft PR/handoff. **First and only immediate dispatch.** |
| **E1** | Mathematically equivalent small **four-library** CPU comparisons: SciPy QMC, PyOptEx, OApackage, OpenTURNS | **DESIGNED / NOT_DISPATCHED** | E0 evidence; optional dependency feasibility; respect different criterion and design semantics | Version-pinned apples-to-apples scalar and structural checks; honest BLOCKED instead of silent substitutions. |
| **E2** | Torch CPU shared-factorization vs full-recompute vs real CUDA batch rank-two exchange and information-matrix kernels | **DESIGNED / NOT_DISPATCHED** | M0 importable torch-only spine + M1/M2 contracts + independent CPU correctness; Scientist authorization to run GPU | End-to-end synchronized time, peak GPU memory, solution-quality parity and error audit, not an isolated GEMM speedup. |

**Never launch E1/E2 as part of E0.** This is an experiment program, not a mandate to implement all 91 PyDOE exports. E0 may start in parallel with M0 only using an isolated branch/worktree and no edits to M0 files. If the Engineer is already busy implementing Issue #1, keep E0 **READY / NOT_STARTED** until the Engineer explicitly CLAIMs it; do not presume a GitHub issue automatically executes code.

## 2. E0 — executable minimum experiment specification (CPU, finite, read-only baseline)

### 2.1 Trust boundary, package lock and reproducibility

- Frozen upstream *source/publisher*: [PyDOE 1.6.0](https://pypi.org/project/pydoe/1.6.0/), source commit `866d3f411942cb662af8054562dc1f7e07372401`, wheel `pydoe-1.6.0-py3-none-any.whl`, **expected SHA-256** `5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec`. Source provenance and wheel digest are **publisher metadata until bytes are actually downloaded and hashed**.
- Use an **isolated virtual environment** (prefer Python 3.13 if installed; otherwise compatible Python >=3.11), separate from TorchOED runtime. Record `sys.version`, OS, CPU, `platform`, `numpy`, `scipy`, `torch` versions, `pydoe` distribution version and `pydoe.__file__`, `pip freeze`. Never introduce PyDOE/NumPy/SciPy as TorchOED runtime deps.
- Download wheel once and verify SHA-256 before import. A bounded retry is allowed for a transient download failure; after that, report **BLOCKED_NETWORK** with exact stderr and zero false `PASS` results. If a verified local wheel is already available, use it; do **not** substitute an unpinned different PyDOE version, GitHub HEAD, or a source transcription as a wheel parity result. Do not modify upstream source.
- Python deterministic seed `20261029`; `np.random.default_rng(20261029)` where APIs accept a Generator, global `np.random.seed(20261029)` only in isolated source-specific comparisons, `torch.Generator().manual_seed(20261029)` for torch. Preserve each engine's native RNG; exact cross-engine random bitwise parity is **not** required.
- All CPU scalar comparisons use `float64`. First verify exact shape/indices/labels and finite status; for well-conditioned numerical formulas initially use `atol=rtol=1e-9` but **tighten if known exact**. For singular/ill-conditioned cases, assert explicit rank/exception/diagnostic status, never silently pass after default ridge or type conversion. Count/IDs/exception type: exact integer or string match.
- Local scripts/tests can live in a **scoped experiment-only path** such as `experiments/reference_e0/`. No `src/torchoed`, package install files, `sae.yaml`, `AGENTS.md`, `docs/DEVELOPMENT_PLAN.md`, `docs/SCIENTIFIC_MENTOR_RESEARCH.md`, `docs/ARCHITECT_ECOSYSTEM_SURVEY.md` edits, and no dependency changes to the M0 runtime. If M0 writes overlapping files, stop and coordinate.
- Strict statuses: `PASS_MATCH` (measured original wheel and oracle agree); `FAIL_MISMATCH` (actually executed original wheel and diverged from a properly specified expected property); `EXPECTED_EXCEPTION` (declared negative case is reproducible); `SOURCE_SEMANTIC_DIFFERENCE` (upstream’s documented mathematical target differs from the independent reference, with **actual wheel observation**); `BLOCKED_NETWORK/INSTALL`; `NOT_RUN`; `INCONCLUSIVE`. A **source-only** result must never be labeled real package parity.

### 2.2 Case matrix: small, independent, falsifiable

| Case ID | Actual PyDOE invocation / frozen inputs | Independently specified math oracle and acceptance | Common misinterpretation |
|---|---|---|---|
| **P0 wheel** | `import pydoe`; `importlib.metadata.version("pydoe")`; compute wheel SHA | Installed distribution 1.6.0; local SHA equals published metadata; exact import path / environment captured | PyPI webpage evidence ≠ wheel locally installed |
| **P1 full-rank** | `small_composite_design(n=4, center=(2,2))` | Count output, build **full** quadratic feature basis with intercept, linear, squares, pairwise products. **Corrected after E0 PR #3:** expect `(20,4)` with full quadratic feature **rank 15 of 15** for actual PyDOE 1.6.0. Keep the former **rank-12** `D=ABC` reference as a *different negative control*, not a valid wheel expectation. Preserve original failed-hypothesis evidence | N≥p does NOT imply model identifiability |
| **P2 shell** | `doehlert_shell_design(num_factors=k,num_center_points=1)`, k=1,2,3,4 | Source-math predicts k=1 **IndexError**, and k=2/3/4 output row count **6/10/15**, rank 2; compare source predictions to wheel exact rows and docs claim 7/13/21 | A source mathematical issue is not already a measured wheel defect |
| **P3 digital net** | `niederreiter_sequence(num_points=16,dimension=1,n_bits=b)` for b=3,5,8,30 | Source Hankel GF(2) generator rank **2**, observed first-coordinate distinct counts **4** (or record exact disparity). Independent GF(2) elimination on generator polynomial verifies rank, not `torch.SobolEngine` parity | Valid Niederreiter theory ≠ PyDOE implementation |
| **P4 sparse point cloud** | `doe_sparse_grid(n_level=L,n_factors=2,grid_type=g)`, L=0,1,2,3; g=clenshaw_curtis, gauss_patterson, chebyshev. Also `sparse_grid_dimension(7,2)` and `(8,2)` | Verify `grid_type` makes **no coordinate difference** in frozen implementation (strict array comparison), expected counts 1/5/13/29 for small levels and 705/705 formula for 7/8; compare against **separate true Smolyak weighted quadrature** but label it a different API target | Same node count ≠ same nodes/weights/quadrature order |
| **P5 Sobol** | `sobol_sequence(n=3,d=2,scramble=False,skip=0,use_pow_of_2=True/False)`; also `(n=4,d=2,skip=1,use_pow_of_2=True)` | With pow2=True expect 4 samples, False 3; SciPy direct QMC and torch unscrumbled first four reference; skip1+pow2 must record **actual exception class** or actual returned output; no assertion of cross-library scrambled seed equivalence | SciPy backend failure ≠ all PyDOE calls fail |
| **P6 exchange** | `fedorov`, `simple_exchange_wynn_mitchell`, `modified_fedorov`, `detmax` on small **distinct-ID** candidate sets; D, degree=1, n>=p; use fixed finite sets from Mentor §22 plus one easy well-conditioned design | Record full output coordinates, rank, normalized objective and selection identity only if uniquely mappable to original candidates. Compare full single-swap neighborhood with exhaustive mathematical oracle; record whether returned design is one-swap local. Compare original routines to **their own** stop rule first, not automatically to ideal classic algorithms | Returned coordinates alone may not prove candidate IDs for duplicates/near-duplicates; mathematical counterexample may be unreachable from source’s own initializer |

**Prioritize P0 → P1/P2/P3/P5, then P4/P6.** Keep only a handful of bounded (tiny) cases. `detmax` is not automatically required to reproduce Mitchell's *true* multi-step excursion; report **source's restricted neighborhood** separately. For `modified_fedorov` with duplicated physical rows or near-equal candidates, do not invent IDs from fuzzy coordinate mapping.

### 2.3 Output schema, evidence and exit gates

Provide one line per actual case with fields:

```json
{
  "case_id": "P1",
  "upstream": {"distribution":"pydoe","version":"1.6.0","wheel_sha256_verified":false},
  "environment": {"python":"...", "numpy":"...", "scipy":"...", "torch":"...", "cpu":"..."},
  "input": {"n":4,"center":[2,2],"seed":20261029},
  "actual": {"called_original_pydoe":false, "shape":null, "exception":null},
  "oracle": {"kind":"independent_torch_math","model":"full_quadratic","expected_rank":12},
  "comparison": {"status":"NOT_RUN", "max_abs_error":null, "tolerance":{"atol":1e-9,"rtol":1e-9}},
  "limitations": ["Example only; no full API option coverage."]
}
```

Before moving to `PASS_MATCH`, `actual.called_original_pydoe` must be **true**, a verified wheel hash and version must be present, and the independent oracle must not simply call the PyDOE function to compute its own expected answer. **Negative/source defect experiments count as valid reproduced findings**, not general TorchOED failures.

**Minimal Engineer output:** `experiments/reference_e0/README.md`, `run_reference.py` (or equivalently compact test module), `results/summary.md`, `results/cases.json` plus machine-readable environment/command record; if binary wheel downloaded, **do not commit wheel**. Prefer compact code, no framework scaffolding, no GitHub Actions. A Draft PR from isolated engineer branch **only when actual results or an honest BLOCKED record are ready**, with `ENGINEER_HANDOFF` including exact commands, base/main SHA, changed paths, inputs, version/hash, PASS/FAIL/BLOCKED/NOT_RUN, performance NOT_RUN, and non-goals. Architect independently reviews diff and reruns critical small checks before any future merge decision; Engineer NEVER self-merges.

**Exit E0 VALIDATE:** (a) correct sha-pinned wheel installed and actually invoked; (b) ≥4 nontrivial case families with truthful actual results, including ≥1 negative edge, unless an explicitly verified BLOCKED_INSTALL is the only attainable outcome; (c) independent reference and package source clearly separated; (d) no runtime dependency drift or M0 path overlap. If wheel fails to download, E0 ends **BLOCKED, not PASS** and offers a bounded rerun on already-authorized network-capable CPU.

## 3. E1 external library comparator — design only, not dispatched

- **SciPy QMC**: LHS/Sobol, seed/skip/sample count, OA strength and sampling discrepancy; avoid asserting identical PyDOE/torch stochastic sequences.
- **PyOptEx**: start with a canonical unconstrained, well-conditioned D-opt small finite candidate instance. Verify exact selection size and scores under the *same feature model and normalization*. Only then try mixture, blocking or cost constraints, which are not directly comparable to unconstrained Fedorov.
- **OApackage**: use `oapackage.exampleArray(0)`, `Defficiency()`, `rank()`, `GWLP()` (official documented API), compare actual array, effect-model dimension and normalization to a separately computed mathematical model. Its first example returns D efficiency 1 and rank 2 under its own conventions; do not compare directly with PyDOE's absolute percentage-style `d_efficiency`.
- **OpenTURNS**: `SmolyakExperiment([marginal1,marginal2], level).generateWithWeights()` and `LHSExperiment`. Validate weight sum and polynomial integrals, parameterization and nodes. The marginal `WeightedExperiment` type and quadrature convention must be explicit; do not compare with PyDOE's **unweighted** `doe_sparse_grid` as if the two outputs had same target.

Pin/record exact distributions on first execution; use independent envs if compiled OApackage/OpenTURNS is incompatible. Always report BLOCKED when unavailable, do not grow default TorchOED dependencies. Execute only after E0 Engineer handoff and explicit next-scope acceptance.

## 4. E2 Torch CPU/GPU kernel break-even — design only, not dispatched

Prerequisites: M0 package importable with a torch-only runtime, mathematical criteria and rank-two swap oracle frozen by M1/M2, actual GPU available on existing approved entitlement. Grid **not yet executed**:

| Tier | Candidates m | Selected n | Features p | Independent instances B | Target |
|---|---:|---:|---:|---:|---|
| Tiny | 32 | 8 | 4 | 1 | Scalar oracle and overhead |
| Medium | 512 | 32 | 16 | 8 | Batched shared factorization vs full recompute |
| Stress | 4096 | 64 | 32 | 8 | Tiled candidate swap scoring + memory bound |

Measure **actual end-to-end** candidate scoring and selected-swap quality, not just matmul: warmup, repeated timing, synchronization, transfers, allocations, CPU threads, device, dtype, peak memory, solver failures and rank. Workloads must have identical feature matrices/score definitions and acceptance rules across CPU and CUDA. Baselines: independent slow complete recompute, torch CPU direct, torch CPU batched rank-update, torch CUDA rank-update where available; optional PyDOE CPU only when its criterion and stopping semantics match. Pass/fail correctness before interpreting latency, and do not add CUDA/Triton dependencies/custom kernels. GPU results are `NOT_RUN` until an actual GPU session produces reproducible evidence.

## 5. Coordination and no-conflict gates

- **Scientist** approves scope; Architect specifies reproducible experiments and independently accepts evidence; **Codex Engineer** executes on a claimed isolated branch, opens Draft PR and posts `ENGINEER_HANDOFF`; never self-merges.
- M0 Issue #1 remains the only feature-spine development authority. **E0 is a separate experiment-only task**, not a shortcut to skip M0. Avoid duplicate issues, conflicting claims, concurrent writes to the same files, unapproved changes to visibility or external paid services.
- Codex running in a separate local/Desktop session **cannot be inferred** from a GitHub issue; status is only verified through a CLAIM/handoff/branch/PR or provided terminal evidence.
- Once M0 PR appears, Architect can decide whether E0 waits or proceeds independently; no automatic expansion of the M0 implementation or accidental merging of E0 artifacts into M0.

**ADR (updated 2026-10-29):** `E0_ENGINEER_CLAIM_AND_HANDOFF_RECEIVED`; `DRAFT_PR_3_INITIAL_REVIEW_WITH_P1_ERRATUM`; `E1_NOT_DISPATCHED`; `E2_NOT_DISPATCHED`; `M0_ISSUE_1_OPEN`; `NO_ENGINEER_MERGE`. Original READY status was a dated snapshot.



---

## 6. E0 final Chief Architect evidence gate: scoped science accepted, PR merge held (2026-10-29)

> Final independent review of [Draft PR #3](https://github.com/JohnnyZPiggy/TorchOED/pull/3) at frozen PR head `8a4f2e8cc51653594387196c9bac95279f1bbe3d`, base `e6c0052adb2bff559c625a8319f89b061c971e9f`. **Do not merge**. E0 scientific findings are accepted on the documented evidence; a small follow-up correction is required to promote the experiment runner to an ongoing clean regression baseline. This review does NOT approve M0 code, GPU, external E1/E2 experiments or automatic merges.

### 6.1 Scope and integrity verification: PASS

- PR remains OPEN/DRAFT/UNMERGED; 11 changed files, all under `experiments/reference_e0/**`. No M0 runtime, setup, CI or original research code changed by the PR.
- GitHub `results/SHA256SUMS` lists **10** other files; a **separately implemented JavaScript SHA-256** hash over the real GitHub-fetched UTF-8/LF content matched **10/10** expected values. The manifest itself is the 11th file, not self-listed. The independent implementation passed the standard SHA-256 "abc" known-answer test.
- Exact `run_reference.py` and `test_reference.py` Git blob IDs were identical at tested commit `f5e9fde69fb31e5e4e9691158f213fe181c38260` and current reviewed PR head `8a4f2e8cc51653594387196c9bac95279f1bbe3d`. Their SHA-256 checksums independently match both manifest and `results/environment.json` `committed_source_sha256`.
- Reviewed entire ~405-line `run_reference.py`, all tests (~22 lines), remaining per-file patches and results manifests. 39 case records parse, with **25 PASS_MATCH, 1 FAIL_MISMATCH, 2 EXPECTED_EXCEPTION, 11 SOURCE_SEMANTIC_DIFFERENCE**; this matches the handoff, summary and recorded runner log.
- P0 Engineer reports original wheel downloaded/locally hashed, original `pydoe==1.6.0` actually imported, **76** installed Python files matched the wheel and **six** critical frozen Git blobs matched the source commit. Independently checked **six pinned Git blob SHA strings match the corresponding 76-file evidence map**. However original wheel archive itself was **not available to this Architect execution environment**, so its on-disk bytes and independent install were **NOT_RUN locally**, not falsely labeled Architect-reexecuted.
- `results/commands.json` records finite Windows execution commands/timeout and **intentional comparator exit 1** due P1 old expected hypothesis; `results/tests.txt` records **3 passed**, exit 0. Those exact Windows processes were **NOT_RUN again by Architect**; their tracked logs/source are consistent. The three Python unit tests cover math helpers (determinant, GF(2), Smolyak), **not** the full 39-case original-wheel regression.

### 6.2 Independent mathematical audit: PASS for measured evidence; generalization limited

- **P1 corrects prior Mentor conjecture**: direct independent NumPy/SymPy reconstruction from actual wheel's 20×4 P1 rows gives full quadratic design `20×15` rank **15**, exact Gram determinant **18,014,398,509,481,984**, and clear smallest singular values (~1.748). The wheel's cube is `D=AB`; prior `D=ABC` **different** cube gives rank **12**, a legitimate counterexample only for that other design. This scientific hypothesis failure **must remain visible** in raw original P1 case and handoff; the correct *future* P1 regression expects rank 15.
- **P2/P3/P4 replay**: from each stored original array, independently rechecked 3 shell rank/count cases (k=2..4, rank2), four Niederreiter integer/GF(2) recurrence first-axis cases (4 unique each), 12 sparse point counts and four exact cross-option coordinate equalities. P2 k=1 IndexError and P5 skip ValueError independently verified as **recorded upstream exceptions**, without claiming a fresh Architect upstream import.
- **P6 eight independent exact-coordinate index/score reconstructions**: eight output designs map to unique original candidate IDs; separate numeric Gram determinants reproduce stored normalized scores within float64 tolerance and reproduce closed 1-swap optima. Four methods on each of 2 small pools gave `global_gap=0` in submitted data, matching independently evaluated scores and exhaustive finite candidate checks; no claim of general global optimality.
- Independent local CPU helper tests re-implemented exact 1D D variance, GF(2) rank+digital first-axis recurrence, and rational-weight Smolyak polynomial integration; mathematical replica suite **PASS**. It is not a byte-for-byte rerun of Codex's Windows pytest file, nor actual installed-wheel execution.

### 6.3 Evidence-label and regression-cleanup requirements (no loss of original results)

**E0 SCIENTIFIC ACCEPTANCE: PASS_WITH_CORRECTED_HYPOTHESIS.** Four-plus nontrivial actual wheel families, negative cases, provenance documentation and scope gates all met on Engineer evidence reinforced by independent source/data/math review.

**PR #3 MERGE GATE: HOLD, NOT APPROVED.** Before considering this experimental branch as a reusable **clean regression**:
1. **P1 add a separately named corrected oracle** for the frozen actual wheel: positive-case rank 15/15 and exact `det(FᵀF)>0`, retaining the old P1=FAIL_MISMATCH original run unchanged as historical falsification. Prefer a small `P1-corrected` test and a new evidence/run summary rather than retroactively changing old `cases.json` or rewriting original logs.
2. **P4 four derived `grid_type` comparisons** are calculated from the 12 actual PyDOE calls, but they are appended directly with `actual.called_original_pydoe=true` without a per-row wheel/source provenance object. Mark them explicitly **DERIVED_FROM_ACTUAL_CASES** with the three constituent case IDs each, and do not count them as four extra direct PyDOE calls. Thus there are **39 evidence records, 35 provenance-marked original-call/installation records, four derived cross-option comparisons**. The original 39 remains intact; append supplement/correct presentation rather than suppressing findings.
3. **P6 clarify** that `PASS_MATCH` in `run_reference.py` checks ID legality and numerical D-score agreement; `one_swap_local_exact`, `global_gap` are reported diagnostics rather than `PASS` predicates. The two current pools are globally best; this is not an algorithm-level performance guarantee. No need to expand E0 to harder or paid tests.
4. Preserve original wheel SHA/Source SHA, exact tested commit, script hashes, environments, finite commands, negative cases, corrected research §25, and the transparent GPU NOT_RUN/COLD-WARM caveat. Maintain experiment-only paths. If any runner code changes, rerun finite CPU comparisons on same pinned wheel and record a **new tested commit**. Do not edit the original raw evidence to manufacture PASS.

**Scientist/Architect boundary:** This is a completed *E0 evidence audit*, **not** permission to merge Draft PR #3. E1, E2, M0, CUDA, release and deployment remain separately gated. Architect will verify any Engineer follow-up before changing MERGE_GATE=HOLD.


---

## 7. E0-CORRECTION-01 final independent Chief Architect acceptance (2026-10-29)

**Reviewed object:** Issue [#2](https://github.com/JohnnyZPiggy/TorchOED/issues/2), exact-head [Draft PR #3](https://github.com/JohnnyZPiggy/TorchOED/pull/3) at `1d59ae5da6c694230aba2e95f217b517cea692c8`. Base `e6c0052adb2bff559c625a8319f89b061c971e9f`; Engineer tested source commit `778db21ad78e770b4cd1ae0e703a6cb8a4ec0e2b`. PR OPEN/DRAFT/UNMERGED. **Formal status: `E0_CORRECTION_01_ACCEPTED`; `E0_SCIENTIFIC_DELIVERY_PASS`; `PR3_MERGE_NOT_AUTHORIZED`.** A separate Scientist approval is required for any merge; this acceptance is not a GitHub APPROVE action.

### 7.1 Independent integrity and scope gates — PASS

- Changed files total **21**: original 11, append-only new 10, all under `experiments/reference_e0/**`; no M0 changes, runtime dependency drift or GitHub Actions.
- Independently computed SHA-256 over **9/9** new files in `results/SHA256SUMS_SUPPLEMENT_01` with a separately implemented hasher self-checked against SHA-256 of abc. The manifest itself is the tenth new file. All nine hashes match the recorded actual GitHub file bytes.
- Independently compared exact Git Blob SHA IDs of **11/11** original files to the frozen original archive (`8a4f2e8cc51653594387196c9bac95279f1bbe3d`); all are unchanged, including original `results/SHA256SUMS` (previous 10/10 hashes verified). Historical `results/cases.json` still has **39** rows: 25 PASS_MATCH, 1 FAIL_MISMATCH, 2 EXPECTED_EXCEPTION, 11 SOURCE_SEMANTIC_DIFFERENCE. The historical P1 FAIL remains and its original runner exit 1 has NOT been relabeled.
- All four original and new Python scripts (`run_reference.py`, `test_reference.py`, `supplement_01.py`, `test_supplement_01.py`) have byte-identical content/Git Blob IDs at the exact tested-source commit and reviewed PR head. New script/test SHA-256s match both the new manifest and supplement evidence map.
- Engineer rehashed the **actual local original PyDOE 1.6.0 wheel** against expected SHA-256 `5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec`, and reported installed 76 Python source-file byte checks. These are reproducible *Engineer-executed* results in Windows Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, Torch 2.14.1+cpu, pytest 9.0.3. Architect did NOT download or execute that wheel on its own host.

### 7.2 Function-level supplement gates — PASS

- **P1 corrected positive:** Exactly one new original-wheel `pydoe.small_composite_design(4,center=(2,2))` call, separately logged. Actual points exactly match the original 20×4 wheel output. Architect independently rebuilt full quadratic 20×15 features and, using separate exact-integer Bareiss elimination, obtained **rank 15** and `det(F.T@F)=18014398509481984`. Distinct `D=ABC` mathematical negative control is **rank 12, determinant 0**; explicitly not a PyDOE defect. Old FAIL is preserved.
- **P4 honest provenance:** Each of four `P4-L{0..3}-grid-type` rows is labeled `DERIVED_FROM_ACTUAL_CASES` and `called_original_pydoe=false`, and references the three actual source-case IDs for its level. Architect checked all 12 referenced ID records and equality of corresponding arrays. **Historical 39 records = 35 direct/import provenance records + four derived comparisons**, not 39 separate package calls.
- **P6 semantics:** New independent helper and four supplement tests enforce that PASS means only **legal candidate IDs and normalized D-score agreement**. Altering `one_swap_local_exact`, `global_gap` or `exhaustive_global_D` does not alter PASS; illegal IDs or wrong scores fail. Architect independently replayed **8/8** original candidate-ID selections and D-score values. The two tiny instances do not certify general global optimality.

### 7.3 Actual CPU logs and what was not run

- Engineer’s new bounded commands and immutable logs: focused original-wheel supplement **exit 0** (180 s max); original oracle tests **3 passed / exit 0** (60 s); supplement tests **4 passed / exit 0** (60 s). Exact command argv and tested-source SHA are recorded in `results/supplement_01_commands.json`; independently verified new log SHA-256s are in the 9/9 manifest. Original 39-case runner was **NOT_RERUN** because its historical source/outputs were unchanged.
- **NOT_RUN by Architect**: independent physical Windows PyDOE wheel install/command rerun. **NOT_RUN project-wide**: complete PyDOE 91-function parity, TorchOED M0 implementation, E1/E2, GPU/WARM/CUDA, performance benchmark, PR merge.

**Gate conclusion:** `ENGINEER_HANDOFF_CORRECTION_01_VERIFIED` and `E0_EXPERIMENT_ACCEPTED_WITH_EVIDENCE_BOUNDARIES`. P1/P4/P6 supplements need no further correction under existing scope. **Leave PR #3 Draft/Open and Issue #2 Open until separate Scientist decision; do not merge, start GPU, or open new Issue/PR.** Recheck exact head before any future permitted merge.
