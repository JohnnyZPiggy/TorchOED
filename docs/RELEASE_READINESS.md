# M5-A release-readiness audit

The table below is updated to 0.9.0 at `b4e5c2ddff308aec916d72750ed9901ca922bfa2`; the M5-A text and base/SAE lines are historical.

Base: `67921b7d9c1a65e82d16f7768ae8672e69e3c3e0`.
SAE: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`.
This is a gap checklist, not release approval. Alpha 0.1.0 has an experimental API.

| Area | Evidence/status | Remaining gap |
|---|---|---|
| Installation/packaging | VERIFIED historically: editable, wheel and outside-checkout import in [M3-B1 evidence](../reports/m3b1/ENGINEER_HANDOFF.md); src/ layout, setuptools >=77, Python >=3.11, torch >=2.0 metadata inspected at historical base | Fresh install/build NOT_RUN in M5-A; dependency environments were nonhermetic. Minimum Python/torch and clean offline/network installs not certified. |
| License | VERIFIED by inspection: root MIT license, project metadata agrees | Distribution license inclusion relies on historical wheel evidence, not a fresh build here. |
| Support matrix | CPU-only [CI matrix](../.github/workflows/ci.yml): Linux + Windows × Python 3.11/3.13, macOS × Python 3.13, clean wheel/sdist, `min-torch` (Python 3.11, torch 2.2.0 CPU), and `docs`. [Metadata](../pyproject.toml): Python >=3.11, torch >=2.2. | Other Python/torch combinations and GPU/TPU are not certified. M3 requires CPU float64. |
| API/contracts | VERIFIED by inspection: 42 root exports in [torchoed.__all__](../src/torchoed/__init__.py); M1/M2/M3 contracts linked in README | Experimental stability; inline annotations/stubs and static type-check acceptance are absent. No broad PyDOE compatibility. |
| Regression coverage | CPU-only CI as in the support-matrix row above ([see badge](../README.md)) | Acceptance is limited to the current CPU suite, not exact or global optimality. |
| Usability | README describes all merged milestones; one tiny existing-API example with assertions | See PR ENGINEER_HANDOFF for exact tested SHA, real smoke output and current result. |
| Release process | Tags `v0.7.0`/`v0.8.0`/`v0.9.0` and TestPyPI publications through [testpypi.yml](../.github/workflows/testpypi.yml); latest evidence in [#518](https://github.com/JohnnyZPiggy/TorchOED/issues/518). Real PyPI: NOT_RUN. | Real PyPI and the GitHub Release remain Scientist-gated; no release authorized here. |

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

## Current TestPyPI rehearsal: 0.9.0

[Issue #518](https://github.com/JohnnyZPiggy/TorchOED/issues/518) records G3 4/4: tag `v0.9.0` on `ac666ee`; [TestPyPI run 44881722587](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44881722587) succeeded; [green main CI run 44870365554](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44870365554) includes macOS. The clean-venv TestPyPI install passed 42/42 exports, the quick start and 26/26 industrial-guide blocks. Versions 0.7.0 and 0.8.0 were earlier rehearsals ([#356](https://github.com/JohnnyZPiggy/TorchOED/issues/356) records 0.8.0). The API remains experimental; real PyPI and the GitHub Release remain Scientist-gated under the checklist below.

## Real PyPI checklist (Scientist)

1. Register a pending trusted publisher on pypi.org for project `torchoed`, owner `JohnnyZPiggy`, repository `TorchOED`; choose and record the workflow filename and environment name.
2. After that setup is confirmed, the Engineer adds a `pypi` publishing job/environment mirroring `.github/workflows/testpypi.yml`, using those exact publisher names and the production endpoint. This Issue adds no publishing job.
3. The Engineer creates the tag for the version the Scientist names (currently 0.9.0) and the GitHub Release only after an explicit Scientist instruction; neither is authorized by this checklist.
4. After an authorized upload, create a fresh venv, install CPU torch, install the version the Scientist names (currently 0.9.0) with `python -m pip install --no-deps torchoed==<version>` from pypi.org, and verify `python -c "import torchoed; print(torchoed.__version__)"` prints that version outside the checkout.

TestPyPI publication is a separate rehearsal; it does not grant production publication approval.
