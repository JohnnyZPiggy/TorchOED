# M5-A release-readiness audit

Base: `67921b7d9c1a65e82d16f7768ae8672e69e3c3e0`.
SAE: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`.
This is a gap checklist, not release approval. Version remains experimental 0.0.1.

| Area | Evidence/status | Remaining gap |
|---|---|---|
| Installation/packaging | VERIFIED historically: editable, wheel and outside-checkout import in [M3-B1 evidence](../reports/m3b1/ENGINEER_HANDOFF.md); src/ layout, setuptools >=77, Python >=3.11, torch >=2.0 metadata inspected at base | Fresh install/build NOT_RUN in M5-A; dependency environments were nonhermetic. Minimum Python/torch and clean offline/network installs not certified. |
| License | VERIFIED by inspection: root MIT license, project metadata agrees | Distribution license inclusion relies on historical wheel evidence, not a fresh build here. |
| Support matrix | VERIFIED historically on recorded Windows CPU environment | Linux/macOS, other Python/torch versions, GPU/TPU NOT_RUN; no support or performance claim. M3 requires CPU float64. |
| API/contracts | VERIFIED by inspection: 13 root exports; M1/M2/M3-A/M3-B1 contracts linked in README | Experimental stability; inline annotations/stubs and static type-check acceptance are absent. No broad PyDOE compatibility. |
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

[M5-A bounded validation handoff](../reports/m5a/ENGINEER_HANDOFF.md).
