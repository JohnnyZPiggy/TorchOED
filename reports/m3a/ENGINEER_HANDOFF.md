# ENGINEER_HANDOFF — Issue #9 / M3-A

**M3A_F1_F12_CPU_ORACLES_PASS / TORCH_ONLY / ARCHITECT_REVIEW_PENDING**

Base `5abafc65ae464a36c6d830136902830f254b9a3e`.
Tested source `4de56c1b7d60c76aaa18a443935240a60b197f60`.
Branch `feat/9-m3a-fedorov-exchange`. The final evidence head is identified by
the Issue comment/PR, avoiding self-reference in this file. Runtime, tests,
runner and pyproject remain byte-identical to tested source; hashes recorded
in validation.json. SHA256SUMS inventories **all changed files except itself**,
canonical LF Git bytes. The Draft PR Files changed view is the complete diff.

CLAIM: https://github.com/JohnnyZPiggy/TorchOED/issues/9#issuecomment-6506845618
Read full Issue9/M3 gate, Mentor §§11–14 and exchange/stability §22, project
AGENTS/sae.yaml/.sae entrypoint and M2 API. Unchanged numerical contract/roadmap
and actual previously fully read SAE constitution/COLD/WARM/task/security/lean
content reused; Engineer reread at immutable SAE pin
d84942e4890d9837ae0e6cbb379b85c3a95aba84 (protocol-reading.json).
Historical gate/metadata non-dispatch labels are superseded only by this scoped
user/Issue dispatch. Live conflict check found no M3 CLAIM/branch/PR; M0/M1/M2-A
PR4/6/8 merged. E0 Draft3 remained at1d59ae5da6c694230aba2e95f217b517cea692c8.

## Implemented slice

Only public `fedorov_exchange`: supplied initial exact IDs, no repeats,
unbatched finite CPUfloat64 model features; deterministic best-improvement
D-optimal 1-swap. Composes merged exact_information/d_optimality and native
triangular solves, tiled Gamma, full direct Gram/D verification of **every**
legal neighbor. Global verified maximum anchors tie tolerance; lexicographic
prefix rechecks select smallest outgoing position then incoming ID. Order
preserved, RNG untouched, no discrete-selection autograd claim.

No copied PyDOE source, inverse, ridge, inverse of delete-only state, top-K
screening, automatic initialization or hidden replication. Original M1/M2
runtime/test files unchanged except package export addition; historical reports,
LICENSE, roadmap, frozen gate/math/research and E0 unchanged. README, scoped
coverage/current SAE dispatch, API docs and validation evidence updated.

## Numeric boundaries and certificate strength

See docs/M3_API.md for exact defaults and error policy. All candidates must pass
merged M2-A numerical SPD policy; conservative pivot rejection with nonzero QR
diagonal is **inconclusive**, never a claim of exact rank deficiency. Zero
numerical QR pivots diagnose singular swaps numerically, not rigorously.
Gamma<=0/nonfinite or ratio/direct disagreement on a directly SPD trial stops
inconclusive. No Gamma clipping. Finite inputs whose trial Gram overflows are
also diagnosed, not accepted.

Gain heuristic g=128*eps64*p*max(1,|Dold|,|Dtrial|)*max(1,cond2(M),cond2(Mtrial)).
Accept only directly verified delta>g. Marginal positive delta<=g cannot block
a clear best move when its lower heuristic gain exceeds all marginal upper
gains; otherwise stop inconclusive. These are **heuristic numerical bounds**,
not rigorous intervals. No result claims LOCAL_1_SWAP_CERTIFIED/global optimum.
LOCAL_1_SWAP_NUMERICAL, NUMERICAL_INCONCLUSIVE, MAX_ITER and NO_LEGAL_SWAP are
distinct. The final-state complete-scan flag resets after every accepted move.

No unlimited n*m tensor/history: explicit max_elements caps shapes and a
conservative major-tensor tile workspace estimate; caller tile width is reduced
if needed and rejected if even one column cannot fit. This is not a measured
peak-memory guarantee or a bound on native linalg workspace/Python allocation.
Full direct verification intentionally favors correctness over speed; there is
no performance assertion. Direct tie rechecks are separately counted.

## Actual CPU validation

Windows11, AMD Ryzen7 4800H 8c/16t; Python3.14.6, torch2.14.1+cpu, pytest9.0.3.
Fresh task venvs use system-site-packages to reuse installed dependencies:
**nonhermetic**. No installed/called PyDOE. Mandatory package requirements
verified exactly torch>=2.0; version stays0.0.1, no release claim.

Actual argv/cwd/durations/exits/timeouts: validation.json,
explore-commands.json, repair1-command.json. Public copies redact private
paths/hostname and normalize whitespace; original raw logs retained locally.
TASK_ROOT is the dedicated task directory on designated D-drive Dropbox;
BASE_ENV is existing D-drive Anaconda; D_BUILD_TEMP is a separate owned D-drive
temporary directory, selected using earlier Windows build-lock evidence.
All builds/caches/results remain on D:, no large C:/WSL scratch/global changes.

Preparation: base Python `-m venv --system-site-packages <TASK_ROOT>/venv`;
venv `-m pip install --no-deps --no-build-isolation -e .` (120s).
Final committed-source command, from checkout:

```text
<TASK_ROOT>/venv/Scripts/python.exe tools/validate_m3a.py --output <TASK_ROOT>/validation-final --temp <D_BUILD_TEMP>
```

Five bounded subprocesses: pytest180s, wheel build/venv/install120s each,
outside-checkout wheel import60s. All exits0. The runner additionally emits
four finite tiny independent exhaustive diagnostic cases. The wheel import
exercises all prior APIs and the new search; eleven exports checked.

| Gate | Actual result |
|---|---|
| COLD / fresh CPU WARM | PASS: analytic Gram error0, disk roundtrip exact, owned file absent |
| F1/F2/F6/F9 | PASS: log4 improvement, saturated Gamma1/4 despite singular deletion, best move [2,1], no inverse |
| F3/F4/F5 | PASS: equal features at different IDs legal, repeats/invalid initial design reject |
| F7/F8 | PASS: position/ID tie independent of tiles/RNG; all legal Gamma ratios vs exact Fraction determinants |
| F10 | PASS: max_iter0/1, no-neighbor labels and strict monotone accepted history |
| F11 | PASS: all six subsets; local determinant1 vs global6561/2500=2.6244 |
| F12 | PASS: singular/near-singular/overflow/disagreement controls; no false exact certificate |
| Full regression pytest | 54 passed,0 failed/skipped,3.84s; unchanged M0/M1/M2 regression tests included |
| Editable/wheel/independent-env outside import | PASS; actual bounded commands/logs retained |
| GPU/Actions/paid/M3-B/benchmarks | NOT_RUN / not authorized |

tests/oracle_exchange.py is independent stdlib Fraction/itertools Gram and
elimination, with all distinct-ID subsets for m<=8 and exact neighborhoods;
no optimized ratio/search API generates its expected values. Determinants and
IDs use exact equality, well-conditioned Gamma/log scores abs/rel1e-9.
oracle-diagnostics.json retains complete tiny subset tables and scan metrics:
F1 final/global4 (logdet error0); F11 final1/global6561/2500; near-singular F12
NUMERICAL_INCONCLUSIVE with original IDs; integer6 final/global18. Equality to
the global optimum in some fixtures is not a general guarantee.

Final wheel SHA-256:
ac72b152c07cc342e492bb3be6eea873aa391524666fd5845e121e57a27446e0.
Wheel retained locally, not published as a release.

## Real failures, repairs and residue

Initial uncommitted EXPLORE:53 PASS/1 FAIL. Independent integer6 oracle expected
best swapped determinant10, but search stayed at4 because an irrelevant
machine-precision positive tie caused global inconclusive. Original failure
trace/JUnit retained (explore-pytest.log/explore.xml). One meaningful repair
separated marginal gains as documented; bounded targeted14 tests passed.
An additional prepublication audit strengthened live-tile workspace accounting
and added a constrained-budget regression. Intermediate ff85dcb validation
passed; final source4de56c1 rerun passed all54. Intermediate and final console
logs are distinguished; validation.json/JUnit refer to **final** tested source.

During evidence assembly, the source-identity guard caught an accidental read
of intermediate validation JSON after the workspace change. Corrected the
assembler path to validation-final, then hashes passed; no numerical source or
fixture was changed to bypass the guard. Observed tool error noted in summary.

warm-local.json records exact roundtrip hash/cleanup; warm_probe.py preserves
the executed source (copy to a fresh owned D-drive directory with scratch/ and
evidence/ subdirectories to reproduce). Scratch and build-temp contain zero
files. Editable/final/intermediate environments (~12.4MB each), wheels/logs and
small build/test metadata intentionally retained for reproduction; residue.json
records counts/bytes. No claim of deleting the checkout/dependencies.

Limits: one Windows CPU/torch environment, inherited dependencies; numerical
conditioning/rank/heuristic tolerance policy, no rigorous certificate,
cross-platform/float32/batch/GPU performance or autograd-through-selection claim.
No M3-B, modified Fedorov, DETMAX, k-swap, multistart or new PyDOE call.

Next: Architect independently reviews source, full-neighborhood/best-tie
semantics, numerical uncertainty and workspace policies, exact exhaustive
oracle, manifest and installation evidence. Issue remains open; Scientist
decides eventual merge. Engineer does not approve or merge own PR.
