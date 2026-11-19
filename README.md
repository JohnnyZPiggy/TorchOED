<div align="center">

# TorchOED

**Optimal experimental design in PyTorch — with a certificate on every design.**

[![CI](https://github.com/JohnnyZPiggy/TorchOED/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyZPiggy/TorchOED/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://johnnyzpiggy.github.io/TorchOED/)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-0.8.0-orange)](https://test.pypi.org/project/torchoed/)
![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-ee4c2c)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Documentation](https://johnnyzpiggy.github.io/TorchOED/) ·
[Industrial guide](https://johnnyzpiggy.github.io/TorchOED/industrial/) ·
[Coming from PyDOE](https://johnnyzpiggy.github.io/TorchOED/pydoe/) ·
[Capability matrix](docs/CAPABILITY_MATRIX.md) ·
[Roadmap](docs/VISION.md)

</div>

TorchOED chooses which experiments to run. It covers classical and industrial design of experiments — D-, A- and I-optimal designs, blocking, split-plot, mixtures, screening and response-surface designs, power and diagnostics — in one tensor API, and it plugs into any PyTorch model.

## Why TorchOED

| | |
|---|---|
| 🎓 **Certified designs** | Every optimal design comes with a **proved lower bound on its efficiency** (equivalence-theorem duality): you know how far it can be from the best possible design. You can also certify a design produced elsewhere. |
| 🏭 **Industrial DOE, in Python** | Blocking, split-plot / hard-to-change factors, mixtures and mixture-process designs, covariates, fractional factorials with aliasing, Plackett–Burman, central composite, Box–Behnken, definitive screening, power analysis and design diagnostics. |
| 🔥 **Any PyTorch model** | `jacobian_features(model, X)` turns a nonlinear model or a neural network into a locally optimal design through autograd; `next_batch` proposes the next experiments in an ask/tell loop. |
| ⚡ **GPU scale** | The exchange search runs on GPU: on 10⁵ candidates in 5 factors it finished in about 3 minutes while PyDOE timed out (see [Benchmarks](#benchmarks)). |

Runtime dependency: `torch>=2.2` only. Python ≥ 3.11. Tested on Linux, Windows and macOS.

## Install

```sh
# Pre-release from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple torchoed==0.8.0

# or from source
git clone https://github.com/JohnnyZPiggy/TorchOED.git
pip install -e TorchOED
```

## Quick start

A D-optimal 7-run design for a full quadratic model in two factors, with its efficiency certificate:

```python
import torch
from torchoed import full_factorial, polynomial_features, optimal_design

C = full_factorial((3, 3)).double() - 1          # 3×3 grid of candidate settings
F = polynomial_features(C, 2)                     # full quadratic model
result = optimal_design(F, 7, criterion="D",
                        generator=torch.Generator().manual_seed(0))
print(result["ids"].tolist())                     # chosen candidate rows
print(result["efficiency_lower_bound"])           # proved lower bound on D-efficiency
```

Switch `criterion` to `"A"` or `"I"`, add `fixed_ids=` to augment runs you have already done, or restrict the candidates with `feasible_candidates` for constraints.

### Blocks and split-plots

```python
import torch
from torchoed import full_factorial, optimal_design, polynomial_features

C = full_factorial((3, 3)).double() - 1
F = polynomial_features(C, 2, include_bias=False)   # blocks absorb the intercept
result = optimal_design(F, 8, blocks=[4, 4],         # two days of four runs
                        generator=torch.Generator().manual_seed(0))
print(result["ids"].tolist())
```

Split-plot designs use `whole_plots=`, `hard=` and `eta=`; see the [industrial guide](https://johnnyzpiggy.github.io/TorchOED/industrial/) for split-plot, mixture, screening and power examples.

## What's inside

| Area | Functions |
|---|---|
| Optimal designs | `optimal_design` (D, A, I, compound and alias-penalised; blocks, whole plots, covariates, augmentation), `next_batch` |
| Classical constructors | `full_factorial`, `ff2n`, `fractional_factorial`, `fold_over`, `plackett_burman`, `central_composite`, `box_behnken`, `definitive_screening`, `lhs`, `sobol` |
| Mixtures | `simplex_lattice`, `simplex_centroid`, `mixture_vertices`, `scheffe_features`, `mixture_process_features` |
| Models and features | `polynomial_features`, `effects_coding`, `feasible_candidates`, `jacobian_features` |
| Evaluation | `design_diagnostics` (correlation, VIF, prediction variance, power), `word_length_pattern`, D/A/E/c/G/V/I criteria |

The live status of every capability — criteria, augmentation, certificates, diagnostics, GPU and external references — is tracked in the [capability matrix](docs/CAPABILITY_MATRIX.md).

## Benchmarks

**Scale vs PyDOE 1.6.0** (D-optimal, full quadratic, random candidates; one Colab host with a Tesla T4; seconds per restart; [full report](reports/w5-crossover/run2-summary.md)):

| factors | candidates | PyDOE (fastest of 3) | TorchOED CPU | TorchOED GPU |
|---|---|---|---|---|
| 3 | 10³ | 4.5 s | **0.7 s** | 1.0 s |
| 5 | 10³ | 7.3 s | **0.8 s** | 1.7 s |
| 5 | 10⁴ | 199 s | 41 s | **9.2 s** |
| 5 | 10⁵ | timeout (> 30 min) | timeout | **188 s** |

Where both finished, TorchOED's design was as good or better (higher log-determinant) in 4 of 5 cells; at 3 factors and 10⁵ candidates it was 0.02 lower than PyDOE's `fedorov` (21.75 vs 21.77), reported as measured. Each TorchOED design also carries a certified efficiency floor of about 0.97.

**Against R** (re-scored with one shared metric): FrF2 fractional-factorial catalogue 22/22 and Plackett–Burman 3/3 identical; all rsm central-composite and Box–Behnken references matched; a blocked response-surface case reached a higher D-criterion than AlgDesign `optBlock` (log det 18.03 vs 16.78); the skpr split-plot documentation example matched exactly. Every external run records its exact R and package versions; references that could not be run are reported as such. See [`reports/`](reports/README.md).

## How it is built

TorchOED is developed by one PhD researcher with an AI agent team: an architect agent writes the mathematical gate documents and reviews every pull request at an exact commit, and an engineer agent implements, tests and benchmarks. Every merge needs green CI on Linux, Windows and macOS; every new identity is checked against exact rational or exhaustive oracles; frozen held-out cases are run once a week and reported as-is. The process and its records are public in the Issues and pull requests.

## Status and roadmap

Pre-1.0 (0.8.0 on TestPyPI): the API may still change. Next: space-filling designs, Bayesian I/D designs, power by simulation for GLMs, then analysis (least squares and REML) with a lightweight TorchOED Studio app, Bayesian and minimax designs on GPU, and 1.0 on PyPI. See the [roadmap](docs/VISION.md).

## License

[MIT](LICENSE)
