# E0-CORRECTION-01 — append-only supplement

Canonical dispatch: [Issue #2 comment 6494815655](https://github.com/JohnnyZPiggy/TorchOED/issues/2#issuecomment-6494815655). Existing Draft PR #3 only; **NO_MERGE**. No M0, E1/E2, GPU or performance work.

## Historical archive stays intact

The original 11 files at `8a4f2e8cc51653594387196c9bac95279f1bbe3d`, including the original `results/SHA256SUMS`, are unchanged byte-for-byte. The original ten manifest entries still match, as verified again by this supplement.

Original tested code: `f5e9fde69fb31e5e4e9691158f213fe181c38260`. Original counts remain **39 records = 25 PASS_MATCH + 1 FAIL_MISMATCH + 2 EXPECTED_EXCEPTION + 11 SOURCE_SEMANTIC_DIFFERENCE**. Preserve P1's old FAIL and the original logs/commands. The legacy comparator's documented exit 1 remains expected historical behavior and is not reclassified as an all-pass run.

The original four P4 aggregate rows contain an ambiguous `called_original_pydoe=true` flag. This supplement supersedes only that interpretation: **39 historical records = 35 direct-function/import-provenance rows + 4 derived comparisons**, not 39 independent API invocations.

## New verified evidence

Tested supplement source: `778db21ad78e770b4cd1ae0e703a6cb8a4ec0e2b`. Script/test byte hashes and Git blob IDs are in `results/supplement_01_cases.json`; original and new Python sources were verified against that commit before execution.

- **One new actual-wheel function call:** `pydoe.small_composite_design(4, center=(2,2))`. Fresh local wheel SHA-256 equals `5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec`; version/import path verified; all 76 installed Python files still equal wheel bytes. No install or download repeated.
- **P1-corrected PASS_MATCH:** X shape 20x4; independently built quadratic F shape 20x15; float64 SVD rank **15**; exact rational determinant of integer Gram **18014398509481984**. This adds a positive regression without relabeling historical P1.
- **Different mathematical negative control:** D=ABC cube with the same axial rows/centers has rank **12**, exact Gram determinant 0. No original-wheel call; not a PyDOE failure.
- **Four P4 provenance corrections:** `evidence_type=DERIVED_FROM_ACTUAL_CASES`, aggregate `called_original_pydoe=false`, with three explicit historical source IDs each. Arrays are independently read from the original twelve direct-call records; no new sparse-grid API calls are claimed.
- **Eight P6 interpretations:** PASS means candidate-ID legality and normalized D-score agreement. `one_swap_local_exact`, `global_gap` and `exhaustive_global_D` are separate diagnostics, not inputs to PASS or a general optimality proof. A new test deliberately changes all three diagnostics and confirms PASS is unchanged; illegal IDs and wrong scores are rejected.
- Focused command exit **0**; legacy oracle tests **3 passed**, exit 0; new tests **4 passed**, exit 0. All finite commands and actual output hashes are recorded separately.
- Legacy runner unchanged; no new full 39-record wheel rerun was needed. Only the focused corrected P1 calls the wheel again. Existing oracle tests were freshly rerun. The prior deterministic seed label is context; focused inputs and mathematical controls use no RNG.

Environment reused: Windows, Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, torch 2.14.1+cpu, PyDOE 1.6.0, pytest 9.0.3. The same D: task-owned venv and wheel remain available. No GPU-WARM claim: prior GPU attempt failed before calculation and its cleanup was confirmed; this supplement allocates no sessions.

## Reproduce the focused check

Use the existing E0-only venv and verified local wheel; do not modify global installations. Commit the new source before running because the script verifies source bytes against HEAD. Choose a **new output path** for another run; the supplement refuses to overwrite prior results.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
# $python and $wheel point to the existing E0-only venv and retained verified wheel.
& $python experiments/reference_e0/supplement_01.py --wheel $wheel --out "$taskRoot/supplement_01_repeat.json"
& $python -m pytest -q -p no:cacheprovider experiments/reference_e0/test_reference.py
& $python -m pytest -q -p no:cacheprovider experiments/reference_e0/test_supplement_01.py
```

Run focused check with a 180-second subprocess limit and each pytest command with a 60-second limit. Exact actual command paths are normalized to `<E0_TASK_ROOT>` in `results/supplement_01_commands.json`; full fresh stdout/stderr is retained in the three new log files.

New artifacts only: `supplement_01.py`, `test_supplement_01.py`, this document, `results/supplement_01_*`, and `results/SHA256SUMS_SUPPLEMENT_01`. The new manifest covers the nine new deliverables except itself; original manifests remain untouched. Final Git blobs are checked against both manifests before handoff.

The previous task-owned venv/cache residue remains as already reported; no deletion was attempted in this correction cycle. No credentials, wheels, environment binaries, private paths or production dependencies are committed. No original-source change, merge or self-approval.

