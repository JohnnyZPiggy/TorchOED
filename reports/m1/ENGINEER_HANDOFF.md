# ENGINEER_HANDOFF — Issue #5 / M1

`M1_F_L_S_P_CPU_ORACLES_PASS / TORCH_ONLY / ARCHITECT_REVIEW_PENDING`

Base: `0dfbc41923e9f9b5dfdb82bde03691415ccf6f3d`.
Tested source: `21d747aa6ea2b9a4bf82a6ce87770df7b0ee6923`.
Branch: `feat/5-m1-constructors`. Final evidence commit is the PR head, identified
by the Issue handoff/PR rather than a self-referential SHA inside this file.
Runtime, tests, runner and pyproject remain byte-identical to tested source;
validation.json supplies their SHA-256s. SHA256SUMS inventories all changed files
except itself, using canonical LF Git bytes. Full final diff is the PR Files
changed view; implementation.diff preserves base→tested-source diff specifically.

CLAIM: https://github.com/JohnnyZPiggy/TorchOED/issues/5#issuecomment-6503865359
Frozen SAE: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`.
Full Issue #5, §9, roadmap, plan and project entrypoints read. Immutable protocols
previously fully read are reused; Engineer protocol reread. Protocol hash inventory
is retained in protocol-reading.json. Live state refreshed before CLAIM and push:
M0 #4 merged, main unchanged, no competing M1 claim/PR; E0 #3 Draft head remains
`1d59ae5da6c694230aba2e95f217b517cea692c8`. Historical §9 architecture-only status
is superseded for this narrow dispatch by explicit Scientist/Issue authorization.

## What changed

Five exported APIs only: integer full factorial, signed ff2n, random LHS, native
Sobol wrapper, degree1/2 polynomial features. Native torch primitives, no copied
PyDOE implementation. Modules, tests, fixture, API docs, README, scoped five-row
coverage update and current SAE dispatch references changed. M0 reports, LICENSE,
research history, numerical contract and release roadmap remain unchanged.
No E0 branch/files changed. Version stays 0.0.1; no release claim.

## Actual environment and validation

Windows AMD Ryzen 7 4800H, 8 cores/16 logical processors; Python 3.14.6,
torch 2.14.1+cpu, pytest 9.0.3, setuptools 83.0.0, pip 26.1.2.
Fresh task venvs reuse installed torch/pytest through system-site-packages:
**not hermetic**, not tested on other versions/platforms. No PyDOE installed.
Production mandatory dependency checked as exactly torch>=2.0.

Actual commands/cwd/timeout/exit/durations are in validation.json, with raw logs
and JUnit. Private absolute paths are replaced with placeholders; mapping:
TASK_ROOT is the dedicated D-drive task directory adjacent to checkout;
WORKSPACE is the Scientist-designated D-drive Dropbox project; BASE_ENV is the
existing D-drive Anaconda install; D_BUILD_TEMP is a separate task-owned D-drive
temporary directory. No large C-drive/WSL temporary data was created.

Executed preparation: base Python `-m venv --system-site-packages <TASK_ROOT>/venv`;
venv Python `-m pip install --no-deps --no-build-isolation -e .` (120s bound);
`-m pytest -q` (180s bound). Final committed-source command, from checkout:

```text
<TASK_ROOT>/venv/Scripts/python.exe tools/validate_m1.py --output <TASK_ROOT>/validation-final --temp <D_BUILD_TEMP>
```

| Gate | Actual result |
|---|---|
| COLD / fresh CPU WARM | PASS; exact Gram error0, disk roundtrip hash and cleanup in warm.json |
| F1–F4 | PASS: exact IDs/order/corners, singleton, preallocation rejection |
| L1–L4 | PASS: exact strata, n1, replay/global RNG isolation, explicit mode rejection |
| S1–S5 | PASS: exact known points/count/skip, seeded scramble replay, zero/invalid policies |
| P1–P6 | PASS: scalar order/bias/ranks/batch, gradcheck, positive/negative controls |
| pytest | 25 passed, 0 failed/skipped, 4.27s (pytest.xml) |
| wheel build/install/outside-checkout import | PASS, separate fresh venv, all five APIs exercised |
| float64 / float32 feature max error | 0 / 2.2649764730431343e-08; tolerance 1e-9 / 1e-6 |
| GPU, Actions, paid services, benchmarks, M2/M3 | NOT_RUN / not authorized |

Rank checks use torch SVD default threshold `max(m,p)*eps*sigma_max`; independent
Fraction elimination verifies exact Gram ranks and determinants. No eigenvalue
acceptance criterion is needed. Gradcheck: eps1e-6, atol1e-5, rtol1e-3.
Exact IDs, strata and rational determinant use equality, not float tolerances.

P6 immutable source artifact SHA-256:
`34ef4a2784ef6f5c528032377d6c0830f1a3bb986fbb21404a724a48a99a458f`.
Fixture records source commit/path, reusing actual E0 PyDOE coordinates; **no fresh
PyDOE call**. Actual20×15 rank15/det18014398509481984 versus separate D=ABC
mathematical20×15 rank12/det0. See test_P6_actual_e0_and_separate_negative.

Built wheel SHA-256:
`92817fbce02992e23eab20f86fa3282fafb0b7586660e8ed2f38389a3f3219ed`.
Wheel retained locally; raw build/install/smoke logs and source hashes published.

## Failures, cleanup and limits

Two initial editable builds failed. The first reported Windows temporary-directory
cleanup locks (WinError32) in tool output; the second retained raw trace reports
a generic pip wheel build failure without the underlying exception. No numerical
test ran in either failed install. Third build succeeded using D_BUILD_TEMP without global changes.
One committed-source collection failure exposed accidentally removed package
exports during documentation preparation; repaired in tested source commit and
all final gates rerun. Failed traces retained, not relabeled PASS.

Owned roundtrip file is absent. Dropbox scratch and D_BUILD_TEMP contain zero
files. Task venvs, wheel, logs and a small pip cache are deliberately retained for
reproduction (see residue.json); no claim of deleting the whole task workspace.
Device forwarding is untested outside CPU. LHS supports random only and bounded
floating resolution; Sobol generates on CPU then transfers, exact-n only;
polynomial features support real float32/64 degree1/2 only. Allocation budget is
an output element bound, not a backend workspace guarantee. No broad PyDOE
equivalence, optimal-design algorithm or performance improvement is asserted.

Next: Architect independently reviews numerical semantics, fixture provenance,
source/manifest and installation evidence; Scientist decides eventual merge.
Issue remains open; Engineer does not approve or merge this PR.
