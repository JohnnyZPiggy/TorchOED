# ENGINEER_HANDOFF — Issue #7 / M2-A

**M2A_A1_A12_CPU_ORACLES_PASS / TORCH_ONLY / ARCHITECT_REVIEW_PENDING**

Base `d718b912f08caa59d6008e8f276b4145cc153918`.
Tested source `9619fb24eb8898ad667b1c1d013ba57e8f0023bc`.
Branch `feat/7-m2a-info-criteria`. Final head is identified by the Issue handoff
and Draft PR (avoiding a self-referential SHA here). Runtime/tests/validation
runner/pyproject remain byte-identical to tested source; validation.json records
SHA-256s. SHA256SUMS covers **all changed files except itself**, canonical LF Git
bytes. The PR Files changed view is the complete final diff/inventory.

CLAIM: https://github.com/JohnnyZPiggy/TorchOED/issues/7#issuecomment-6505065430
Full Issue #7, M2_IMPLEMENTATION_GATE, numerical §§2–6, roadmap, AGENTS and
sae.yaml read. Frozen SAE pin d84942e4890d9837ae0e6cbb379b85c3a95aba84:
actual constitution/Engineer/COLD reread; previously fully read immutable
role/WARM/task/security/lean material reused, hashes in protocol-reading.json.
Current user/Issue dispatch supersedes historical M1-only metadata and M2
architecture-only gate only for this slice. No competing CLAIM/branch/PR at
dispatch; M0 #4/M1 #6 merged, E0 #3 Draft unchanged at
1d59ae5da6c694230aba2e95f217b517cea692c8.

## Scope and numerical policy

Adds only exact_information, weighted_information, d_optimality, a_optimality,
e_optimality. Native torch gather/matmul/cholesky_ex/cholesky_solve/eigvalsh;
runtime torch only, no copied upstream code. Public export additions, small API
docs/README/coverage and current dispatch metadata accompany the two modules,
independent tests and bounded validation runner. M0/M1 history and existing
implementation files except package exports are unchanged, including immutable
E0 fixture, LICENSE, roadmap and frozen math/gate specifications.

See docs/M2_API.md for exact signatures and domains. Exact IDs, explicit repeats,
raw versus normalized event semantics and probability weights remain distinct.
No batch broadcasting; data errors identify the first invalid batch. Raw n0
returns zero; normalized n0, empty weighted design or empty criterion batch
reject. Input/selected/output element caps checked before allocation/linalg.

For M, tau=64*p*eps*max|M|. Symmetry residual must be <=tau, then both triangles
are averaged. D/A require Cholesky info0 and squared diagonal pivots>tau; no
ridge/inverse. The pivot safeguard is conservative and order dependent, not a
condition-number certificate: some true SPD but poorly scaled inputs reject
as numerically singular/ill-conditioned rather than claiming reliable scores.
E rejects eigenvalues<-tau, clips [-tau,0] to numerical zero. This explicitly
distinguishes numerical PSD from exact mathematical PSD. Repeated-minimum E
gradient and PSD/clipping-boundary gradients are NOT_RUN/nonunique/undefined.

## Actual environment and commands

Windows11, AMD Ryzen7 4800H, 8 cores/16 logical processors; Python3.14.6,
torch2.14.1+cpu, pytest9.0.3 (CPU float64 first, float32 separately).
Fresh task venvs use system-site-packages to reuse installed torch/pytest;
**not hermetic**. No PyDOE installed or freshly called. Mandatory installed
production requirements verified as exactly torch>=2.0. Version stays0.0.1.

Exact argv/cwd/timeouts/durations/exits in validation.json and
explore-commands.json; original raw logs retained locally, public copies redact
private paths/hostname and normalize trailing whitespace. Placeholders:
TASK_ROOT=owned D-drive task directory, WORKSPACE=Scientist-designated D-drive
Dropbox project, BASE_ENV=existing D-drive Anaconda, D_BUILD_TEMP=separate
owned D-drive build-temp (chosen using prior M1 Windows-lock evidence).
No large C-drive/WSL scratch or global setting changes.

Preparation: base Python `-m venv --system-site-packages <TASK_ROOT>/venv`;
venv Python `-m pip install --no-deps --no-build-isolation -e .` (120s),
then exploratory pytest (180s). One targeted repair rerun:
`<TASK_ROOT>/venv/Scripts/python.exe -m pytest -q tests/test_m2a.py`, 15 passed
in3.46s. Final committed-source validation, from checkout:

```text
<TASK_ROOT>/venv/Scripts/python.exe tools/validate_m2a.py --output <TASK_ROOT>/validation --temp <D_BUILD_TEMP>
```

The runner bounds pytest180s, build/install/venv120s, outside import60s, fails
immediately on nonzero exit and records failure before raising. It builds a
wheel and installs it in a second fresh task venv, imports outside checkout,
checks all ten public exports including five M2-A functions. All final exits0.

## CPU Oracle results

| Gate | Result |
|---|---|
| COLD / fresh CPU WARM | PASS; analytic Gram error0, owned disk roundtrip and cleanup |
| A1/A2/A3 | PASS: raw diag2, normalized I; exact repeat policy; weighted I |
| A4/A5/A6/A7 | PASS: diagonal/correlated scalar formulas; singular D/A reject, E0; indefinite rejects |
| A8/A9/A10 | PASS: event scaling, invalid inputs/size guard, literal matched-batch matrices and error indices |
| A11 | PASS: float64 D/A/simple-eigenvalue E gradcheck; information features and normalized weights also checked |
| A12 | PASS: actual E0 rank15/det18014398509481984 vs separate D=ABC rank12/det0, D/A negative rejection |
| Full pytest | 40 passed,0 failed/skipped,3.50s; includes unchanged M0/M1 regression tests |
| Editable/wheel/external import | PASS; raw logs, JUnit and wheel hash retained |
| GPU/Colab, Actions, paid, M3, benchmarks | NOT_RUN / not authorized |

Independent expected values are literal scalar/2×2 mathematics, hand-written
feature formulas and Fraction Gaussian elimination reused from the unchanged M1
oracle helper. No self-generated expected result. A12 coordinates reuse
tests/fixtures/e0_p1.json with original E0 artifact SHA-256
34ef4a2784ef6f5c528032377d6c0830f1a3bb986fbb21404a724a48a99a458f.
Full positive integer Gram matches exactly and Cholesky succeeds; logdet matches
log(18014398509481984). Exact rank/determinant independently confirm SVD rank
using default max(m,p)*eps*sigma_max. Actual vs mathematical negative provenance
remains distinct; no new PyDOE call or broad upstream parity.

For M=[[2,1],[1,2]], measured float64 errors D/A/E:
2.220446049250313e-16 / 2.220446049250313e-16 /0; float32:
1.9834087483516782e-08 /7.947285962650597e-08 /0.
Declared abs/rel tolerances1e-9 and1e-5 respectively. Integer matrices/IDs/Gram
and exact rational determinant use equality. Gradcheck uses eps1e-6,
atol1e-5/rtol1e-3 for criteria, documented native defaults for information maps.

Built wheel SHA-256:
b4d53d0a2d18b8693182d42cfb965c586fd75ab54410d7d3739abed805017917.
Wheel retained locally with installation environments, no release/publish action.

## Failure evidence and residues

EXPLORE initially had39 PASS/1 FAIL: A12 exact rank12 Gram unexpectedly passed
native Cholesky and D/A. Original failure trace/JUnit in explore-pytest.log and
explore.xml. One meaningful source repair added the scale-aware pivot guard;
targeted15 then committed-source40 tests passed. No acceptance fixture changed,
no hidden ridge, no failures suppressed. All final validation commands passed.

warm-local.json records exact D-drive roundtrip hash and verified absence after
unlink. warm_probe.py is the exact executed probe source; to reproduce, copy it
to a fresh owned D-drive run directory with scratch/ and evidence/ subdirectories.
Disk scratch and dedicated build-temp have zero files. Approximately12.4MB each
editable/validation environments, small wheel and logs are deliberately retained
on D: for reproduction; residue.json supplies precise counts/bytes. No claim of
deleting the task checkout or dependencies.

Limits: one Windows CPU/PyTorch environment, inherited dependencies; no CUDA,
compile/performance or cross-platform guarantee; conservative pivot-domain
rejection, finite precision PSD clipping and repeated-E gradient limits above.
No nonlinear Fisher, other criteria, optimizer/M3 or efficiency% claim.

Next: Architect independently audits PR source/math, A12 provenance, pivot/PSD
policies, manifest and installation evidence. Issue stays open pending review;
Scientist decides eventual merge. Engineer does not approve or merge own PR.
