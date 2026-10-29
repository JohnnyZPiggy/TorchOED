# E0: original PyDOE 1.6.0 CPU equivalence experiment

Refs #2. Experiment-only; no TorchOED package, M0 files, global environment, upstream source, or SAE entrypoints are changed.

## Identity and outcome

- Base: `e6c0052adb2bff559c625a8319f89b061c971e9f`.
- SAE pin: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`.
- PyDOE publisher/source pin: `866d3f411942cb662af8054562dc1f7e07372401`.
- Original wheel: `pydoe-1.6.0-py3-none-any.whl`, SHA-256 `5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec`.
- Wheel bytes were downloaded once, hashed before installation, installed without dependencies into an E0-only venv, and actually invoked. Every installed PyDOE .py file was compared byte-for-byte with the verified wheel; six relevant files also match exact pinned Git blob IDs.
- Python 3.14.6 / NumPy 2.4.6 / SciPy 1.18.0 / torch 2.14.1+cpu / pytest 9.0.3, Windows, CPU float64, seed 20261029.
- 39 observations: 25 PASS_MATCH, 2 EXPECTED_EXCEPTION, 11 SOURCE_SEMANTIC_DIFFERENCE, 1 FAIL_MISMATCH.
- **P1 corrects the old research prediction:** actual 20x4 design has full quadratic rank **15/15**, not 12/15. Exact Fraction arithmetic gives det(F.T F)=18014398509481984. The independent D=ABC fractional cube gives rank 12 but is a different design. Keep the mismatch; do not call this an upstream bug.
- The comparator intentionally exits **1** while a measured FAIL_MISMATCH remains. Exit 1 here is a scientific finding, not an incomplete run. All P0–P6 results and environment were written; unit tests exit 0.
- GPU, performance, general DOE correctness, complete upstream API parity, E1/E2: NOT_RUN.

## Reproduce

Use a writable task root on the designated D: workspace, outside the checkout. The tested venv inherits existing torch/pytest; this is isolated installation of the baseline, not a claim of a fully hermetic environment. All inherited distribution versions are recorded. Public evidence replaces the actual task root with `<E0_TASK_ROOT>`; exact paths and raw pip freeze remain in local-only task evidence.

PowerShell commands (adjust only the task-root location and existing Python path):

```powershell
$taskRoot = 'D:\E0-reproduction'
New-Item -ItemType Directory -Force $taskRoot, "$taskRoot\scratch", "$taskRoot\wheel" | Out-Null
$env:TEMP = "$taskRoot\scratch"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$taskRoot\scratch\pip"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& 'D:\software\anaconda3\python.exe' -m venv --system-site-packages "$taskRoot\venv"
$python = "$taskRoot\venv\Scripts\python.exe"
& $python -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --retries 1 --timeout 30 --dest "$taskRoot\wheel" pydoe==1.6.0
$wheel = "$taskRoot\wheel\pydoe-1.6.0-py3-none-any.whl"
if ((Get-FileHash -Algorithm SHA256 $wheel).Hash.ToLowerInvariant() -ne '5050cd6925db5c83fb3f6a6318b8f7b59250fa2e6f36e27309db8be128fb71ec') { throw 'wheel checksum mismatch' }
& $python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --ignore-installed --retries 1 --timeout 30 numpy==2.4.6 scipy==1.18.0
& $python -m pip install --no-index --no-deps $wheel
& $python experiments/reference_e0/run_reference.py --wheel $wheel --out "$taskRoot\reproduced-results"
# Expected exit 1: P1 hypothesis mismatch. Inspect all records; do not discard negative findings.
& $python -m pytest -q -p no:cacheprovider experiments/reference_e0/test_reference.py
```

Use finite subprocess timeouts: 180 s comparator, 60 s tests, 30 s pip network operations with at most one retry. The exact final validation commands/exit codes/log hashes are in `results/commands.json`. Commit the experiment source before running: the comparator verifies its own and its tests' bytes against HEAD and records the tested commit. Do not install or run E1 packages.

### Actual environment repair

Initial reuse of inherited Anaconda NumPy/SciPy triggered Intel OpenMP duplicate-runtime Error #15 after the sparse-grid cases. No successful full-run result is attributed to that aborted attempt. No `KMP_DUPLICATE_LIB_OK` override was used. Installed the **same versions**, from official binary wheels, into the E0 venv only; torch/pytest remain inherited. The subsequent full run and final bounded validation completed. `environment.json` records actual module locations, versions, source hashes and sanitized freeze data. Raw local freeze has a retained SHA-256; omitted local build URLs are not portable dependency sources.

## Oracle boundaries

- P1 builds independent full quadratic features and uses torch SVD plus exact rational Gram determinant. A separate D=ABC construction explains the prior prediction.
- P2 measures actual rows/rank and k=1 exception; frozen-source predictions and documented Doehlert row counts are kept distinct.
- P3 derives the x²+x+1 recurrence and Hankel rank over GF(2), using exact rational coordinates. It verifies the frozen recurrence's behavior, **not** general Niederreiter correctness. The b=3 case intentionally exceeds 2**b; b=5/8/30 do not.
- P4 records complete point arrays, counts and equality across options. Its count matches are narrow source-prediction checks; they are not weighted quadrature certification. A separate exact midpoint/Simpson/Boole difference-rule Smolyak example records weights and polynomial integrals.
- P5 calls both real SciPy and real torch independently of PyDOE; unscrambled first points only. No scrambled RNG equivalence claim.
- P6 actually calls fedorov, detmax, simple_exchange_wynn_mitchell and modified_fedorov on an easy pool and the physical §22.3 pool. Degree-1 features are explicitly [1, coordinates]. Exact Fraction determinant scores use normalization det(F.T F/n). Every legal single swap and the finite global optimum are enumerated independently.
- P6 does **not** inject a conditional trap as an upstream starting state. Real methods select their own initial designs. Their particular outputs are local/global on these two pools only; max_iter=20 is a finite cap. Frozen-source simple exchange considers one selected drop, modified Fedorov freezes a sweep pool, and DETMAX's one-add/one-drop is not a genuine multiple-step excursion. This experiment does not establish their contracts for other inputs.
- Numerical equality tolerance: float64 atol=rtol=1e-9, exact equality for IDs/shapes/exceptions/digital points. Conditions, full outputs, warnings and rational scores are retained in cases.json.

## Files and integrity

- `run_reference.py`: original-package calls and clearly separated independent references.
- `test_reference.py`: exact determinant, GF(2), and signed Smolyak oracle checks.
- `results/cases.json`: every input/output/exception/status; negative outcomes preserved.
- `results/environment.json`: verified wheel/source evidence, environment and independent Smolyak data.
- `results/commands.json`, `run.txt`, `tests.txt`: final finite validation and real output.
- `results/summary.md`: concise interpretation.
- `results/SHA256SUMS`: byte integrity of all deliverables except itself.

No wheel or venv is committed. After evidence publication, scratch/pip cache and the disposable venv are removed; the small verified wheel and exact-path evidence are retained outside the checkout. No GPU session was allocated for E0. PyDOE source was read but not modified or copied into TorchOED; references are mathematical formulas. Existing project MIT license is untouched; PyDOE remains its separate BSD-3-Clause development oracle.

