# M5-A release-readiness audit

Base: `67921b7d9c1a65e82d16f7768ae8672e69e3c3e0`.
SAE: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`.
This is a gap checklist, not release approval. Version remains experimental 0.0.1.

| Area | Evidence/status | Remaining gap |
|---|---|---|
| Installation/packaging | VERIFIED historically: editable, wheel and outside-checkout import in [M3-B1 evidence](../reports/m3b1/ENGINEER_HANDOFF.md); src/ layout, setuptools >=77, Python >=3.11, torch >=2.0 metadata inspected at base | Fresh install/build NOT_RUN in M5-A; dependency environments were nonhermetic. Minimum Python/torch and clean offline/network installs not certified. |
| License | VERIFIED by inspection: root MIT license, project metadata agrees | Distribution license inclusion relies on historical wheel evidence, not a fresh build here. |
| Support matrix | VERIFIED historically on recorded Windows CPU environment | Linux/macOS, other Python/torch versions, GPU/TPU NOT_RUN; no support or performance claim. M3 requires CPU float64. |
| API/contracts | VERIFIED by inspection: 12 root exports; M1/M2/M3-A/M3-B1 contracts linked in README | Experimental stability; inline annotations/stubs and static type-check acceptance are absent. No broad PyDOE compatibility. |
| Regression coverage | VERIFIED historical 65-test run, independent exact oracles and manifests at their recorded source | Full suite NOT_RUN for this docs-only change. No current cross-platform acceptance inferred. |
| Usability | README describes all merged milestones; one tiny existing-API example with assertions | See PR ENGINEER_HANDOFF for exact tested SHA, real smoke output and current result. |
| Release process | NOT_RUN: tags, package publication, Actions, release automation | Architect must resolve support/versioning/package checks before any stable release claim; no release authorized here. |

## Reproduction and boundaries

From an installed source checkout: `python examples/cpu_quickstart.py`.
It is a documentation smoke with tiny CPU tensors, not a benchmark, new oracle
campaign or optimality claim. It checks a literal determinant and status/gain
sentinels; two-swap scoring never selects/applies a move.

Reuse historical reports without rewriting them or treating their checksums as
current-doc hashes. Issue #11 is complete; Issue #13 and its probes remain
STOPPED_BY_SCIENTIST. No GPU, Colab, new algorithm, dependency, version change or
self-merge is part of M5-A. Full diff and bounded command evidence are attached
to the single M5-A Draft PR's ENGINEER_HANDOFF.

## ENGINEER_HANDOFF — bounded validation

Tested README/example source: `cd51317acd867d5317a5547a9d062cbe5ac4fd7e`.
The following evidence-only commit does not change the example or runtime;
final exact HEAD is recorded in the PR comment. Branch: `feat/16-m5a-usability`.
No competing open M5-A PR or remote branch was found before implementation.
Previously read immutable SAE Engineer/COLD/WARM/task/lean evidence reused;
current project rules and Issue #16 fully read. This new explicit dispatch
authorizes usability work only and does not resume Issue #13.

Environment: Windows, installed D-drive Anaconda Python 3.14.6,
torch 2.14.1+cpu, setuptools 83.0.0. No environment recreation or install.
The smoke explicitly imports this checkout through subprocess PYTHONPATH=src;
this is not a fresh installed-package check. Child OMP_NUM_THREADS=1 and
MKL_NUM_THREADS=1; no global environment change. From the checkout, the actual
corrected Python launcher was (shell quoting omitted):

```python
import sys, subprocess, os
env = dict(os.environ, PYTHONPATH=os.path.abspath('src'),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
r = subprocess.run([sys.executable, '-B', 'examples/cpu_quickstart.py'],
                   env=env, timeout=60, capture_output=True, text=True)
print(r.stdout)
print(r.stderr)
print('EXIT', r.returncode)
sys.exit(r.returncode)
```

PASS, exit 0; stderr empty. Initial logdet: 2.7725887222397807.
1-swap stop: LOCAL_1_SWAP_NUMERICAL; certificate: NUMERICAL_ONLY.
Two-swap counts: VALID=9, INVALID=0, NUMERICAL_INCONCLUSIVE=0.
Valid gains in enumeration order:
`[0.0, 0.8472978603872032, 0.8472978603872032, 0.8472978603872032, 1.386294361119891, 1.466337068793427, 1.466337068793427, 1.845826690498331, 1.9459101490553132]`.
Literal determinant=16, pair-count=9 and finite/undefined gain assertions pass.
These observations demonstrate example execution only, not new algorithm acceptance.

FAIL (launcher only, corrected): first PowerShell invocation used nested escaped
quotes in Python -c and raised `SyntaxError: unterminated string literal` before
executing the example. Replaced nested code with subprocess PYTHONPATH; no
production/example repair was needed. No remaining example failure.

PASS: `git diff --check`; diff limited to README, this document and one example.
Runtime, tests, pyproject, SAE pin, historical reports and stopped branch are
unchanged. No scratch dataset, environment, build or result binary created;
only the owned D-drive source worktree is retained.

NOT_RUN: full pytest/oracles, wheel/editable reinstall, benchmark, GPU/Colab,
Actions, cross-platform checks, type checker, tag/release/publication.
BLOCKED: no blocker to this usability handoff; stable release readiness remains
limited by the unverified matrix/install/typing gaps above.

Next: Architect independently review exact Draft PR HEAD; no Engineer approval
or merge. Refs #16.
