# ENGINEER_HANDOFF — Issue #11 / M3-B1

**M3B1_K2_SCORING_150_EXACT_CPU_ORACLES_PASS / TORCH_ONLY / ARCHITECT_REVIEW_PENDING**

Base `cb944b0e7cf0a6cd1bfc99743a6e72bc69471ab4`.
Tested source `427674b71647c03f6ec4a8105614c449202a8fcd`.
Branch `feat/11-m3b1-kswap2-scores`. Final evidence head is identified by the
Issue handoff/PR, avoiding self-reference here. Runtime, tests, runner and
pyproject remain byte-identical to tested source (validation.json hashes).
SHA256SUMS covers **all changed files except itself**, canonical LF Git bytes;
the PR Files changed view supplies the full final diff/inventory.

CLAIM: https://github.com/JohnnyZPiggy/TorchOED/issues/11#issuecomment-6509252718
Full Issue11/M3B gate §§1–7, current root rules/SAE entrypoint and M3 API read;
unchanged already fully read Mentor §§11–14/M3 gate/numerical contract reused
after Git byte comparison. Frozen SAE d84942e4890d9837ae0e6cbb379b85c3a95aba84:
Engineer reread, actual previously fully read immutable constitution/COLD/WARM/
task/security/lean reused; protocol-reading.json and reading-provenance.json.
Historic M3-A-only metadata/design-only labels superseded only for this scoped
Scientist/user/Issue dispatch. Live main matched base, PR4/6/8/10 merged, no
competing M3-B1 CLAIM/branch/Issue/PR; own checkout clean. E0 Draft3 unchanged
at1d59ae5da6c694230aba2e95f217b517cea692c8.

## Implemented scope, schema and numerical boundaries

Only public `score_two_swaps(F,ids,*,tile_size=64,max_elements=1_000_000)`:
all legal unordered outgoing position pairs and incoming unselected ID pairs,
CPUfloat64 finite features, unique CPUint64 IDs and initial M2 numerical SPD.
API/schema frozen in docs/M3B1_API.md during EXPLORE before VALIDATE.
Stable position-then-ID lexicographic ordering, including unsorted initial IDs
and equal features at distinct IDs. No input mutation, RNG, winner selection,
accepted/applied endpoint, history, search loop or stop/locality API.

Original M Cholesky + native triangular solves, genuine rank-four determinant
det(I4+J ZᵀZ), J=diag(-1,-1,+1,+1). Every prospective full selected-feature Gram
and D score directly recomputed through merged M2. No inverse, ridge, delete-only
inverse, Gamma clipping, sequential one-swap substitution or copied upstream
implementation. Saturated n=p works without requiring deletion to stay SPD.

Results expose pair positions/IDs, raw ratios, verified direct gains/scores,
tolerances/discrepancies, per-row VALID/INVALID/NUMERICAL_INCONCLUSIVE and reasons.
NaNs are explicit undefined sentinels with non-VALID status, never valid scores.
M2 rejection with numerical QR zero is INVALID (numerical rank diagnostic, not
exact algebraic proof); nonzero QR remains inconclusive, preserving near-singular
mathematical SPD. On direct SPD trials require finite Gamma>0 and log-ratio/direct
gain agreement under the documented condition-aware heuristic. No heuristic
error tolerance is called a rigorous interval. Scoring returns negative/zero/
positive gains without choosing or applying anything. certificate NOT_APPLICABLE.

Count and output bounds checked with Python integers before allocation/linalg.
Retained numeric/index output10C+n, conservative fixed major workspace
2mp+6np+10p²+4m+2n+64 plus per tile pair16p+128; fit/reduce tile width, reject if
one cannot fit. Bounded index generator, no full Cartesian tuple list. Output
itself is O(C) and bounded; this is not a measured peak-memory guarantee for
backend workspace, Python objects or allocator/serialization overhead.

All previous runtime/test/runner files unchanged except package export addition;
all M0–M3-A evidence, LICENSE, roadmap and frozen mathematical/research/gate docs
unchanged. Only new score/oracle/tests/runner/API docs and scoped current metadata,
README, coverage note and this evidence added. No frozen upstream row promoted
for this separate scoring API; 91-row coverage counts unchanged.

## Real environment and bounded commands

Windows11, Ryzen7 4800H 8c/16t; Python3.14.6, torch2.14.1+cpu, pytest9.0.3.
Fresh task editable/wheel venvs inherit installed CPU dependencies using
system-site-packages: **nonhermetic**. No installed/called PyDOE; mandatory
production requires exactly torch>=2.0. Version stays0.0.1, no release.

Actual argv/cwd/exits/durations/timeouts in validation.json and
explore-commands.json; public logs redact private paths/hostname and normalize
trailing whitespace, original raw copies retained locally. TASK_ROOT is the
owned D-drive Dropbox task directory; BASE_ENV existing D-drive Anaconda;
D_BUILD_TEMP a separate task-owned D-drive temp selected using earlier Windows
build-lock evidence. No large C:/WSL scratch or global configuration changes.

Preparation: base Python `-m venv --system-site-packages <TASK_ROOT>/venv`;
venv `-m pip install --no-deps --no-build-isolation -e .` (120s), exploratory
pytest180s. Final committed-source validation from checkout:

```text
<TASK_ROOT>/venv/Scripts/python.exe tools/validate_m3b1.py --output <TASK_ROOT>/validation --temp <D_BUILD_TEMP>
```

Five bounded subprocesses: pytest180s, wheel build/venv/install120s each,
outside-checkout wheel import60s. All exits0. The runner also emits finite150-pair
exact diagnostics and the six-subset rational trap. Wheel smoke verifies twelve
exports and exercises new scoring outside checkout in a second fresh task env.

## Scoped B-oracle matrix — no fabricated full-search acceptance

| Gate | Actual result |
|---|---|
| COLD / fresh CPU WARM | PASS: analytic Gram error0, owned disk roundtrip/hash/readback/unlink/absence |
| B1/B2 | PASS: all four one-swaps81/100, initial1; simultaneous pair6561/2500; all six subsets exact |
| B3/B5 | NOT_RUN / DESIGN_ONLY: Modified Fedorov sweep and DETMAX excursion not implemented |
| B4 | PASS: add-one/drop-one endpoint identity changes at most one membership, not simultaneous two |
| B6 | PASS: seeded7421 small integer k1 determinant identity, all legal exact ratios, no new k1 search |
| B7 | PASS: total m11, selected n6, incoming5 →150 pairs, all150 VALID, exact Fraction/Bareiss cross-check |
| B8/B9/B10 | PASS: saturated, equal features/distinct IDs, enumeration/cardinality/order across tiles and RNG isolation |
| B11 search stopping/monotonicity/certificates | NOT_APPLICABLE; coverage/no-search-surface checks pass without claiming search behavior |
| B12 | PASS: singular, near-SPD, overflow/nonfinite/pathological ratios, IDs/dtypes/repeats/budget negatives |
| Full pytest | 65 passed,0 failed/skipped,5.83s; unchanged earlier milestone regressions included |
| Editable/wheel/outside import | PASS; full raw logs/JUnit/commands and source hashes retained |
| GPU/Actions/paid/M3-B2/search/benchmarks | NOT_RUN / not authorized |

Independent tests/oracle_two_swap.py builds rational Gram and scaled-integer
Bareiss determinant, cross-checked against unchanged independent Fraction
elimination in oracle_exchange.py. No production output generates its expected
values. oracle-diagnostics.json contains every one of150 pair IDs, exact Gram,
integer determinant/rational ratio and measured score/error, plus trap/boundaries.
Integer/rational/ID comparisons exact; well-conditioned float64 abs/rel1e-9.
Measured max ratio abs error2.7000623958883807e-13; max log-gain error
3.1086244689504383e-15. True singular and mathematically near-singular SPD are
reported separately. No PyDOE call/bitwise parity or global/local optimizer claim.

Final wheel SHA-256:
6bcd3a7996fae5f2cacb79eb6f5b79ae6513f8ed3aa140a695e2cbfcfab831b3.
Wheel retained locally; not published as a release.

## Failures, residue and limits

**No EXPLORE or VALIDATE test/build/install failures in this cycle; zero repair
attempts.** Original successful logs retained; no synthetic failure log created.
Earlier milestones' true failures remain in their unchanged historical reports.

warm-local.json records real D-drive path/hash/cleanup with private prefix
redacted; warm_probe.py is the executed source (copy into an owned D-drive run
directory with scratch/ and evidence/ subdirectories to reproduce). Scratch
and D_BUILD_TEMP contain zero files. Editable/validation environments (~12.4MB
each), wheel/logs and small build/test metadata deliberately retained for
reproduction; residue.json gives exact counts/bytes. No claim of deleting the
checkout or reusable dependency environments.

Limits: one Windows CPU/torch environment, inherited dependencies, numerical
SPD/rank/heuristic policy, undefined-score NaN sentinels require status checks;
no rigorous bounds, batch/float32/GPU/autograd/performance or search claim.
No Modified Fedorov, DETMAX, k>2, k2 optimizer, M3-B2 or new paid service.

Next: Architect independently audits exact-head formula/sign/order, complete150
coverage, Fraction/Bareiss independence, schema/uncertainty/memory policy, manifest
and installation evidence. Issue stays open; Scientist decides eventual merge.
Engineer does not approve or merge own PR.
