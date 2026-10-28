# TorchOED — Initial Development Plan

> **Status:** Initial architecture proposal (2026-10-28). This is a planning document, not an implementation claim.
>
> **Project:** TorchOED | **Repository:** https://github.com/JohnnyZPiggy/TorchOED
>
> **One-line goal:** Reconstruct classical Design of Experiments (DOE) and Optimal Experimental Design (OED) algorithms as genuinely PyTorch-native, batched, CPU/CUDA-capable implementations—not wrappers around PyDOE.

## 0. 项目定位 / Mission

TorchOED 以现代 [PyDOE](https://github.com/pydoe/pydoe) 的公开算法分类和数学定义作为功能参考，重构可复用的 Tensor 数学内核，然后实现 DOE 构造器、OED 准则、精确设计搜索和后续空间填充/自适应算法。

核心主张：

1. **PyDOE is a reference, never a runtime engine.** PyDOE 仅作为开发期对照 Oracle 和算法清单。卸载 PyDOE 后，TorchOED 公开 API 必须仍可运行。
2. **PyTorch-native computation.** 核心数值路径使用原生 PyTorch 张量与线性代数算子，不将 NumPy/SciPy 计算结果包装成 Tensor，不在热路径进行 CPU 往返。
3. **Tensorize the numerical inner loop.** 尽量批量化候选评分、矩阵分解与距离计算；保留适度 Python 外层控制流，不为“无循环”牺牲可读性或内存安全。
4. **Correctness before speed, GPU early.** 从第一个数值里程碑开始验证 CPU/CUDA，先以数学与统计证据确认正确性，再优化时间/显存。
5. **Reuse first.** 优先使用 PyTorch 已实现的成熟底层操作（如 \`torch.linalg\`、\`torch.quasirandom.SobolEngine\`）；不在 V1 自写 CUDA/Triton，也不无谓重造已有算法。
6. **Evidence-driven.** 每个算法记录支持范围、数学不变量、Oracle 差异、CPU/CUDA 测试和性能结果；失败和不可支持情形明确暴露。

### 项目边界

- **V1:** 线性/多项式模型的候选集设计；经典 DOE 构造器；标准信息矩阵准则；离散交换搜索；基础空间填充；单实例和批处理；CPU/CUDA。
- **V1 非目标:** 声称实现所有 PyDOE 接口、全面领先 CPU、非线性贝叶斯 OED、通用自动微分保证、手写 CUDA kernels 或生产级分布式求解。
- **未来可选:** \`cuOED\` 作为独立 CUDA-native 性能工程探索；仅基于 TorchOED 的实际瓶颈启动。

## 1. 上游功能边界与独立性

上游主参考（开发开始时再冻结准确的 tag/commit/SHA、依赖、许可证及可观察 API 行为）：

- [PyDOE unified repository](https://github.com/pydoe/pydoe)
- [PyDOE documentation](https://pydoe.github.io/pydoe/)
- [PyDOE optimal module](https://github.com/pydoe/pydoe/blob/master/pydoe/optimal/__init__.py)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

PyDOE 发布为 BSD-3-Clause。TorchOED 应优先采用独立实现；若复制/改编有版权保护的上游代码，须保留适用的版权/许可/声明。发布前核查 TorchOED 的自有代码和第三方依赖。**当前文档不构成已取得 MIT 许可或完成权属审计的声明。**

不承诺逐比特兼容：随机采样、交换搜索、迭代停止、数值容差和结果排序可能与 PyDOE 不同。兼容性以数学定义、输出约定和明确写出的差异为准。不得自称 PyDOE 官方 PyTorch 版本。

## 2. 数学核心与统一张量契约

### 2.1 模型与信息矩阵

候选点 \`c_i\` 经 feature/model map \`phi\` 得到 \`f_i = phi(c_i)\`，设模型矩阵 \`F ∈ R^{m×p}\`。

- **Exact design:** 选中的行构成 \`X ∈ R^{n×p}\`；默认 \`M = X.T @ X\`（需要时可显式选择归一化）。
- **Approximate design:** 权重 \`w_i ≥ 0, sum_i w_i = 1\`；\`M(w) = F.T @ diag(w) @ F\`，实现应使用 weighted GEMM 而非显式创建 \`diag(w)\`。
- **Heteroscedastic extension (later):** 显式输入权重/协方差，不能在默认独立同方差模型中暗含。
- 默认 D 目标：最大化 \`logdet(M)\`；A：最小化 \`tr(M^{-1})\`；E：最大化 \`lambda_min(M)\`。
- G/I/V/C/S/T 的定义和评估分布/点集在实现前逐项冻结，不能用一个未经验证的“通用方差”函数替代。

### 2.2 Shapes, dtype, device, randomness

| Object | Single instance | Batched instances |
| --- | --- | --- |
| Candidate factors | \`(m, k)\` | \`(B, m, k)\` when supported |
| Model features | \`(m, p)\` | \`(B, m, p)\` |
| Chosen features | \`(n, p)\` | \`(B, n, p)\` |
| Information matrix | \`(p, p)\` | \`(B, p, p)\` |
| Candidate scores | \`(m,)\` | \`(B, m)\` |
| Exchange gains | \`(n, m)\` | \`(B, n, m)\` or bounded tiles |

- Support CPU and CUDA when the operation is genuinely device-compatible; avoid unconditional \`.cpu()\`, \`.numpy()\`, hidden transfers or per-candidate \`.item()\` inside numeric hot loops.
- Numeric validation: start with \`float64\` reference, test \`float32\` separately. Integer/combinatorial constructors may return integer tensors; API must explicitly document output types.
- Explicit \`device\`, \`dtype\`, \`generator\` or seed contract where appropriate. Reproducibility is per supported backend/configuration; never promise identical CPU and GPU random sequences.
- Missingness, NaNs, Inf, invalid dimensions, duplicated points, rank deficiency, and impossible experimental sizes must have documented, tested behavior.
- The model matrix must not be confused with physical factor coordinates: polynomial expansion and categorical encoding are separate transformations.

### 2.3 Numerical stability

- SPD matrices: prefer Cholesky and triangular solves (\`cholesky_ex\`, \`cholesky_solve\` as appropriate), \`slogdet\`; avoid explicit inverse/determinant in production code.
- Near-singular/rank-deficient cases: detect and report; use QR/SVD/eigendecomposition only where mathematically justified. **Do not silently add ridge regularization** and call the resulting criterion unregularized D-optimality.
- Specify conditioning tolerances, monotonic-improvement tolerance, tie breaking, stopping criteria, and acceptance/rejection behavior.
- All criteria use a documented normalization and orientation; do not compare non-equivalent normalizations across different \`n\` or \`p\`.
- Preserve an independent slow reference calculation for small problem sizes.

## 3. Algorithm inventory and staged priorities

This is a **coverage backlog**, not a claim that all algorithms already exist in TorchOED. Algorithm names below are drawn from the current PyDOE public catalogue. A name here does not imply direct API compatibility.

| Tier | Families / representative upstream functions | Primary torch strategy | Relative effort | GPU opportunity |
| --- | --- | --- | --- | --- |
| T0: Core | \`information_matrix\`, \`build_design_matrix\`, \`generate_candidate_set\`, \`criterion_value\`, efficiencies | Matrix multiplication, polynomial feature expansion, batched solves | Medium | High |
| T1: Basic DOE | \`fullfact\`, \`ff2n\`, \`fold\`, \`repeat_center\`, \`random_uniform\`, \`sukharev_grid\` | \`meshgrid\`, \`cartesian_prod\`, tensor indexing, RNG | Low | Low |
| T2: Classical designs | \`fracfact\`, \`fracfact_by_res\`, \`pbdesign\`, \`gsd\`, \`john_three_quarter_design\`, Latin/Graeco-Latin squares, blocking | Combinatorial construction plus tensor transforms | Low–High | Low–Medium |
| T2: Response / mixture | \`bbdesign\`, \`ccdesign\`, Doehlert, \`star\`, \`union\`, simplex-lattice / centroid / axial, extreme-vertices / mixture-process | Formulaic candidate construction, constraints | Low–High | Low–Medium |
| T2: QMC / LHS | \`lhs\`, \`sobol_sequence\`, Halton, Hammersley, rank-1/Korobov, Cranley-Patterson, Faure/Niederreiter | \`SobolEngine\` and native tensor generation when suitable | Low–High | Medium for batches |
| T3: Space filling | \`maximin_design\`, \`minimax_design\`, \`maxpro_design\`, \`nearly_orthogonal_lhs\`, \`oa_lhd\`, \`sliced_lhs\`, \`nested_lhs\`, \`random_k_means\` | Batched/tiled distances, correlations, selection | Medium–High | High for large candidate sets |
| T3: OED criteria | D/A/E/G/I/C/V/S/T, \`a_efficiency\`, \`d_efficiency\` | Factorizations, eigenvalues, prediction-variance kernels | Medium–High | High |
| T4: OED search | \`sequential_dykstra\`, \`simple_exchange_wynn_mitchell\`, \`fedorov\`, \`modified_fedorov\`, \`detmax\`, \`optimal_design\` | Tensorized candidate/exchange gains + outer iteration | High | High with sufficiently large workload |
| T5: Advanced / adaptive | \`fracfact_opt\`, \`fracfact_aliasing\`, \`alias_vector_indices\`, \`hyper_graeco_latin_square\`, Taguchi utilities, sparse grids, Morris/Saltelli/Iman-Conover, definitive/supersaturated, \`sequential_design\`, GP/acquisitions | Specialized discrete math or statistical algorithms | High | Case dependent |

**Coverage tracking rule:** M0 engineer must create an explicit upstream-function inventory with statuses \`UNREVIEWED / DEFERRED / PLANNED / IMPLEMENTED / VERIFIED\`, distinguishing *algorithm*, *criterion*, *helper*, and *public facade*. Do not estimate function count from the table.

### Priority deviation from difficulty order

Implement a handful of simple generators early, but **do not wait for the entire classical DOE catalogue** before shipping the central OED math and exchange algorithms. Prioritize information matrices, criteria and Fedorov soon enough to learn where GPU acceleration helps.

## 4. Fedorov exchange: first optimization target

Let \`A ∈ R^{n×p}\` represent selected rows and \`B ∈ R^{m×p}\` candidate rows. For positive definite \`M = A.T @ A\`, exchanging selected row \`a_i\` for \`b_j\` yields:

\`\`\`text
M_new = M - a_i a_i^T + b_j b_j^T
u_i = a_i^T M^{-1} a_i
v_j = b_j^T M^{-1} b_j
c_ij = a_i^T M^{-1} b_j
det(M_new) / det(M) = (1 - u_i)(1 + v_j) + c_ij^2
\`\`\`

A batched/tiled score matrix can be constructed from solves and matrix multiplies, not \`n*m\` fresh decompositions. Numerical evaluation must account for cancellation and invalid/non-positive gains; fallback can recompute a shortlisted candidate's direct \`slogdet\` before accepting.

- Eligibility mask: legal exchanges only; define duplicate and replacement policies.
- Positive-definite baseline and conditioning requirements explicit.
- Tiles/chunks to avoid unconditional allocation of an \`O(nm)\` matrix.
- Outer iteration stays readable; convergence/monotonicity verified.
- Compare vectorized gains against direct recomputation on tiny independent examples.
- **No performance claim until end-to-end measurements** (candidate scores + selection + updates + sync).

A plain, correct, non-vectorized reference implementation may remain **test-only**, while the shipped numeric path should use torch-native batched scoring.

## 5. Proposed package layout (not yet created)

\`\`\`text
TorchOED/
  README.md
  LICENSE                       # only after author/licensing choice is finalized
  pyproject.toml
  docs/
    DEVELOPMENT_PLAN.md
    ALGORITHM_COVERAGE.md        # generated/maintained during M0
    NUMERICAL_CONTRACTS.md       # M1/M2
    BENCHMARKS.md                # M4+
  src/torchoed/
    __init__.py
    core/{features,information,criteria,validation}.py
    designs/{factorial,response_surface,mixture,lhs,qmc}.py
    optim/{sequential,exchange,detmax}.py
    space_filling/{distance,search}.py
  tests/{unit,oracle,property,reference}.py
  benchmarks/
\`\`\`

This is a suggested module map; engineer may simplify after the first working implementation. Avoid unused scaffolding or generated noise. Public API shapes are **proposals**, to be frozen by M1:

\`\`\`python
import torch
import torchoed

candidates = torch.tensor([[-1.0], [0.0], [1.0]], device="cpu", dtype=torch.float64)
features = torchoed.build_design_matrix(candidates, degree=1)
design, info = torchoed.optimal_design(
    candidates=candidates,
    n_points=2,
    degree=1,
    criterion="D",
    method="fedorov",
)
\`\`\`

No assertion here that this sample already executes or that PyDOE signatures will match exactly.

## 6. Milestones and evidence gates

### M0 — Upstream audit and minimal project spine
- Verify target PyDOE version/tag/SHA and license; enumerate exported APIs and dependency edges, identify already-supported torch primitives and genuine gaps.
- Create concise algorithm coverage table, scientific contract draft, minimal installable package, lean smoke tests.
- **Exit:** package import, torch-only runtime dependency, meaningful small CPU smoke, upstream baseline frozen; no algorithm-performance claim.

### M1 — Tensor constructors and feature maps
- Implement \`fullfact\`, \`ff2n\`, minimal \`lhs\` and/or Sobol reuse, linear/quadratic model matrix construction.
- **Exit:** correct shapes, deterministic invariants, device/dtype tests, explicit random semantics, optional CUDA smoke.

### M2 — Information matrices and OED criteria
- Implement weighted/unweighted information matrix; D/A/E criteria first; then C/G/I/V according to frozen definitions.
- Test singularity, normalization, rank, known small analytical cases, batched \`(B,p,p)\`.
- **Exit:** independent numerical oracle, CPU/CUDA agreement within stated tolerance, no hidden NumPy numeric dependency.

### M3 — Classical exchange optimization
- Start with sequential greedy and Fedorov; then Modified Fedorov and DETMAX once semantics/quality confirmed. Wynn–Mitchell simple exchange follows as a comparison.
- Compare direct tiny reference, exhaustive combinatorial optimum where feasible, independent restarts and quality gap.
- **Exit:** valid designs, reproducible protocol, documented stopping, numerical monotonicity (within tolerance), batched/tiled candidate evaluation.

### M4 — Space-filling and broader DOE
- Add maximin/minimax/LHS variants and high-value factorial/response-surface/mixture families based on actual demand.
- **Exit:** coverage ledger and mathematical properties updated for each implemented family.

### M5 — Expansion toward PyDOE functional coverage
- Bring in specialized low-discrepancy, constrained/combinatorial, Taguchi, sensitivity, sparse-grid, and adaptive algorithms where justified.
- **Exit:** transparent supported/unsupported matrix; no blanket completeness claim.

### M6 — Performance optimization and cuOED decision
- Profile kernel launches, data movement, decomposition, distance scoring, exchange gains and host synchronization; try batching/chunking/compile only where evidence supports it.
- **Exit:** stable workload suite and correctness-vs-runtime/peak-memory results; identify whether any dedicated CUDA-native cuOED kernel is worthwhile.

*Milestones are logical stages, not promises of duration or funding; EXPLORE → VALIDATE and small smoke first. On available CUDA hardware, prioritize inexpensive native GPU probes and scale up only when useful.*

## 7. Acceptance, baselines and benchmark protocol

1. **Math properties:** factorial level coverage; LHS stratification; QMC bounds; information matrix PSD; D/A/E known analytical cases; construction constraints.
2. **Oracle triangulation:** PyDOE optional development comparison + independent slow mathematical reference + closed-form/exhaustive tiny tests. For stochastic/local optimizers, compare distributions, invariants and objective quality, not equal point order or matching random draws.
3. **CPU/CUDA parity:** \`float64\` reference then \`float32\`; log environment, seeds, backend, versions, device, tolerated errors and rank behavior.
4. **Statistical validity:** several independent initializations for local algorithms; report best/median/dispersion and failure cases; separate optimization quality from runtime.
5. **Benchmark fairness:** baseline PyDOE CPU, TorchOED CPU, TorchOED CUDA; equal design size, candidate set, criterion and comparable stopping/quality conditions; include allocation, transfers, synchronization, warm-up and end-to-end wall time. Report peak GPU memory when available.
6. **Scale:** tiny oracle → moderate → stress; test candidate counts, model dimension \`p\`, design runs \`n\`, batch size \`B\`, and singular/near-singular matrices. Include a CPU-competitive baseline; never infer speedup solely from an isolated matmul benchmark.
7. **Honest reporting:** \`PASS\`, \`FAIL\`, \`NOT_RUN\` and \`BLOCKED\` must be distinguished. No synthetic GPU evidence presented as native execution. No silent fallback from CUDA to CPU.

## 8. Engineering workflow / SAE handoff

**Roles:** Scientist (human) defines scope and accepts science; Architect/Mentor proposes contracts, evaluates evidence; Engineer (Codex or another agent) codes on isolated branch and opens PR. Do not merge automatically.

**Lean workflow:** EXPLORE → VALIDATE; reuse-first; focused tests before expensive experiments. CPU/GPU available: try small real CUDA runs early, allow bounded retries, avoid unnecessary approval/price gates. Favor a small executable increment and reviewable diffs, not large empty directory trees or defensive scaffolding.

### First Engineer assignment (M0 only)

1. Inspect \`JohnnyZPiggy/TorchOED\` live; avoid duplicate issues/branches/PRs.
2. Read this plan; independently inspect current PyDOE source docs and source exports. Freeze an exact upstream ref and document discrepancies with this initial inventory.
3. Create exactly one scoped M0 Issue and a working branch; keep changes limited to the minimal package spine, algorithm coverage inventory, smoke tests and developer instructions. No premature full OED implementation.
4. Ensure PyDOE is not a runtime dependency. Add a smoke test that runs with PyDOE unavailable; verify torch-native import and a tiny CPU tensor path.
5. Run tests, record exact commands, versions, results, and any honest \`NOT_RUN\` GPU checks. Open a PR and supply a concise \`ENGINEER_HANDOFF\` with changed paths, evidence, limitations, and suggested M1 contract.
6. **Do not merge** without Scientist/Architect review.

**M0 acceptance:** Minimal installable TorchOED package, pinned upstream reference and complete coverage classification, truthful CPU smoke, no hidden PyDOE/NumPy numeric backend, reviewable PR, reproducible handoff.

## 9. Open architecture decisions

- Functional parity vs public signature compatibility: prioritize semantic parity; freeze APIs after M1.
- Repeated experimental points, no-replacement constraint and weighted designs: explicit per-method choices.
- Default model bases (linear, full quadratic, interactions, categorical encoding) and column ordering.
- How to express G/I/V integration measures and T-model discrimination; upstream conventions must be audited.
- Batched exact search semantics for variable candidate counts (padding/masks vs separate problems).
- Gradient support: meaningful for differentiable continuous criterion paths, not a blanket promise for discrete exchange algorithms.
- Publishing/license/name: choose after code and third-party review, rather than claiming current release readiness.

## 10. Definition of done for V1 (proposed)

V1 means a **usable, tested OED core**, not complete PyDOE parity:

- Native tensor API for feature maps and information matrices.
- D/A/E criteria, reliable rank/conditioning behavior, small independent correctness oracle.
- Working sequential and Fedorov search with documented constraints.
- At least one factorial constructor and one space-filling/QMC constructor.
- CPU and real CUDA correctness evidence on supported hardware, plus fair end-to-end performance baselines.
- No PyDOE runtime dependency; explicit function coverage and limitations; lean docs/tests and license review.

**North star:** “Not PyDOE wrapped in Tensor; DOE/OED algorithms rebuilt around tensor algebra.” 
