# M0 Engineer evidence — CPU only

Refs #1. EXPLORE → VALIDATE completed; **Draft PR / independent Architect review required; NO_SELF_MERGE**.

- Base: `4dc2dd7e6a94a3437bac1df6e76dc4c0585a0728`.
- Tested source: `d7a0b3fa8941555fc55d8d251952c22b8f6d3a13`.
- After testing, precise reuse-strategy wording, LF attributes for evidence logs/manifest and this evidence were updated. Runtime files, tests and pyproject.toml stay byte-identical to the tested source.
- Existing LICENSE and six original project files remain unchanged. E0 PR #3 head remains `1d59ae5da6c694230aba2e95f217b517cea692c8`; no E0 files/branch changed.

| Check | Actual result |
|---|---|
| COLD | Role/repository/git/gh/current Issue/PR/conflict state refreshed; independent feat/1-m0-spine |
| WARM | Existing actual CPU/D: placement/integrity/cleanup evidence reused; fresh M0 package checks below |
| EXPLORE | Editable install exit0; pytest4 passed exit0; pre-commit working source, not a pinned validation claim |
| VALIDATE editable install | Actual no-index/no-deps/no-build-isolation editable install of tested SHA, exit0 |
| VALIDATE pytest | 4 passed, exit0; CPU oracle, metadata, PyDOE absence and torch-only source boundary |
| Fresh wheel environment | Before install, PyDOE and TorchOED absent; new venv inherits existing torch/pytest only by environment policy |
| Wheel build/install | Successful local wheel build and install into fresh venv, both exit0 |
| Installed-wheel import | Imported from that venv's site-packages while running outside checkout, PASS |
| CPU float64 Gram | Shape2x2, output [[3,0],[0,2]], max error0, symmetry/PSD and singular/no-ridge tests PASS |
| Storage | Actual tempfile root on designated D: scratch; 24-byte roundtrip JSON read/hash verified, single file removed |
| Frozen upstream | v1.6.0 tag/commit, actual wheel and sdist SHA-256, selected source/metadata/license byte checks PASS |
| Coverage | 91 distinct root names =89 functions +2 classes including Enum; 17 PLANNED /74 DEFERRED /0 implemented or verified public features |
| Submodule definitions | 11 unprefixed module-level definitions absent from root __all__, separately listed |
| SAE adoption | Minimal pin entrypoints; YAML parsed/pin/CPU scope checked; 12 distinct immutable links verified |
| GPU/Colab/Actions/paid services | NOT_RUN / not started per user scope |
| M1/M2/M3, compile/autograd/performance | NOT_RUN / no implementations or claims |

## Environment and exact commands

Windows11 AMD64; Python3.14.6, torch2.14.1+cpu, pytest9.0.3, setuptools83.0.0, pip26.1.2. See [validation.json](validation.json) for real environment/module location, exact argument arrays/cwd/timeout/exit codes, individual log text+SHA-256, source Git blobs and reading coverage. Public paths are normalized to `<M0_TASK_ROOT>`; local exact-path evidence remains on the designated D: workspace.

Core commands actually run, not suggested/fabricated:

```sh
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -m pytest -q -p no:cacheprovider
python -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir <task-scratch>/wheels .
python -m pip install --no-index --no-deps <built-torchoed-wheel>
```

Install/build/venv commands were bounded at120s; tests/import/storage commands at60s. The installed-wheel CPU command is fully recorded, including its -c program, in validation.json. Tests disable third-party pytest plugin autoload and bytecode writes through process-local environment only. No system settings changed.

The two fresh venvs deliberately inherit the existing torch/pytest so there is no PyTorch reinstall or large C: temporary/cache use. This is **not a hermetic environment** or a claim that NumPy/SciPy are absent from torch's host installation. PyDOE is genuinely absent in both test environments; the M0 package has no NumPy/SciPy/PyDOE imports or declared dependencies. Raw wheel/archive inspection uses only stdlib in a separate audit process.

[commands.log](commands.log) preserves human-readable EXPLORE/VALIDATE output; [audit.json](audit.json) records the name inventory, static dependency boundary, source/archive provenance and adoption checks. [SHA256SUMS](SHA256SUMS) covers each new source/document/evidence file except itself using canonical Git bytes. No wheels, archives, venvs or private user paths are committed.

## Storage, residue and scientific limits

Actual task-owned paths are under `<designated D: workspace>/.sae-m0-20261029/`: `checkout/`, `evidence/`, `scratch/`, `venv/` and `validation-venv/`. The roundtrip temporary file was removed and cleanup verified. Reproducibility assets remain deliberately: venv12382085 bytes, validation-venv12388202 bytes, scratch1952913 bytes (verified upstream sdist +4081-byte TorchOED wheel). These retained directories are **not claimed cleaned**; unrelated E0/WARM data were untouched.

The wheel contains exactly two Python modules: package version metadata and one private Gram helper. The helper is not a public information-matrix, candidate, criterion or solver implementation. No public upstream feature is marked implemented. Only one Python/torch configuration and tiny CPU fixtures were tested. Other Python versions, devices, compile/autograd, performance and full upstream equivalence remain unverified.

Wheel license text equals the unchanged project LICENSE after CRLF normalization; Windows checkout line endings explain the raw-byte difference. Both license SHA-256 values are recorded separately.

Next: Architect review this M0 exact source/diff/evidence, then decide acceptance and a separate M1 dispatch. No self-merge; no automatic continuation to algorithms.
