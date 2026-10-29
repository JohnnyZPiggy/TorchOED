# TorchOED

**M0: installable PyTorch-only package spine.** No public DOE/OED algorithm APIs are implemented yet. This is an independent project; it is not an official PyDOE port or wrapper.

## Install and CPU smoke

Python >=3.11; torch is the sole mandatory numerical runtime dependency. pytest is an optional test extra.

```sh
python -m pip install -e ".[test]"
python -m pytest -q
python -c "import torchoed; print(torchoed.__version__)"
```

For an existing torch/pytest environment, an offline editable install can use `python -m pip install --no-deps --no-build-isolation -e .`. Tests require PyDOE to be absent. M0 validation uses a fresh venv inheriting the already-installed torch/pytest; it is not a claim of a hermetic or NumPy-free torch installation.

The only numerical package code is a **private**, non-public smoke helper: `F.mT @ F`. The CPU float64 fixture `[[1,-1],[1,0],[1,1]]` must produce `[[3,0],[0,2]]`, preserve shape/dtype/device, and satisfy symmetry/PSD checks. It is not the planned public information-matrix API.

## Scope, references and limits

- [91-name inventory](docs/ALGORITHM_COVERAGE.md): root exports include helpers/classes, not 91 algorithms. All public feature implementations remain PLANNED or DEFERRED.
- [Numerical scope note](docs/NUMERICAL_CONTRACTS.md) and [future M1/M2 contract](docs/M1_M2_NUMERICAL_CONTRACT.md): definitions, not M0 algorithm implementations.
- PyDOE1.6.0 is a frozen, optional **development oracle**, never a runtime dependency or numerical backend. See the inventory for wheel/tag/source/license provenance. Existing E0 PR #3 remains a separate experiment.
- GPU, compile/autograd claims, performance, full upstream equivalence and M1/M2/M3 are NOT_RUN / not implemented by M0.
- Original project code is covered by the existing [MIT LICENSE](LICENSE). PyDOE has its own BSD-3-Clause license; no upstream implementation was copied into the package.
- [SAE entrypoint](AGENTS.md) pins V3.3. Engineers submit reviewable branches/PRs and never self-merge.

## 中文说明

当前仅完成 M0 最小工程骨架：可安装、可导入，数值运行依赖只有 torch；没有公开的 DOE/OED 算法、求解器或占位接口。CPU 测试使用内部 Gram 运算和解析小矩阵，不等于实现 M2 信息矩阵 API。PyDOE 只作为开发期参考，不参与生产数值计算。91 个名字的清单不是算法完成率。后续 M1/M2/M3、GPU 和性能实验需要独立范围与真实证据；本轮未执行。工程师不自行合并。
