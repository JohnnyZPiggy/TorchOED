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
