# TorchOED — Scientific Mentor Mathematical Research

> **SAE V3.3 | Chief Architect + Scientific Mentor | Research dossier v1.0 | 2026-10-28**
>
> **Scope:** Mathematical research, upstream source audit, small independent CPU checks, and future research hypotheses. This is **not** an implemented TorchOED feature list, performance benchmark, merged PR, or Engineer authorization.
>
> **Upstream frozen baseline:** PyDOE **1.6.0** (released 2026-10-21), source commit **866d3f411942cb662af8054562dc1f7e07372401**, license **BSD-3-Clause**. Baseline references in Section 16.
>
> **TorchOED planning baseline:** main at start of this study **fc26dfaddfd7d2ca080e72c7ee4f00823b16ab13**, [docs/DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md), and existing [Issue #1](https://github.com/JohnnyZPiggy/TorchOED/issues/1).
>
> **English abstract:** TorchOED is a proposed PyTorch-native reconstruction of classical designs of experiments and optimal experimental design. We group the PyDOE surface into combinatorial constructors, probabilistic and low-discrepancy samplers, feature maps, information matrices, optimality criteria, exact/approximate design search, space-filling metrics, and adaptive designs. The shared numerical backbone is batched feature construction, weighted information matrices, factorizations, solves, candidate quadratic forms, distance kernels, and bounded-memory reductions. This document derives key contracts and Fedorov exchange identities, audits important differences from the frozen PyDOE behavior, records independently checked examples, and distinguishes mathematical proofs from CUDA performance hypotheses.

---

## 0. 科学问题与研究边界

**定位：** 不是“把所有 PyDOE 函数包装成 Tensor”，而是把算法目录重新映射成少量可组合的数学原语，研究其统计含义、输入输出约定、可证明性质和数值限制，再决定什么值得 PyTorch-native 实现。

**严格区分：**

- **THEOREM / PROVED：** 在明示假设下的数学定理与代数恒等式。
- **CPU_CHECK：** 独立 Python/PyTorch 数值实验；不等于 TorchOED 包通过测试。
- **UPSTREAM_AUDIT：** 固定 PyDOE 源码的真实行为；不能想当然推广到以后版本。
- **HYPOTHESIS / RESEARCH_PENDING：** 尚未建立可靠误差界、鲁棒性或 GPU 性能数据。
- **NOT_RUN：** TorchOED 工程实现、原生 CUDA 基准、端到端性能测试。

PyDOE 根包固定提交的公开 **__all__ 包含 91 个名字**，但其中既有算法、准则、辅助函数、类与门面；**91 不等于 91 个独立算法**。实际开发覆盖清单仍由 M0 的算法分类任务维护。本导师文档不是已实现覆盖率表。

研究假设优先基于有限候选集与线性参数模型，必要时扩展非线性 Jacobian、异方差、相关观测误差和贝叶斯/序贯设计。这些场景的准则不能混同。

## 1. 统一数学结构：从设计因素到信息与搜索

给定 m 个物理候选实验点 C∈R^{m×k}，特征映射 f(c;θ)∈R^p，候选特征矩阵 F∈R^{m×p}。注意 **k 是因素数量、p 是模型参数数量**；尤其二次多项式下二者通常不同。

模型 y=f(c)^T β+ε，独立同方差 E[ε]=0、Cov ε=σ²I，则对满列秩的选中设计 X∈R^{n×p}：

$$
\operatorname{Cov}(\hat\beta)=\sigma^2(X^\top X)^{-1},\qquad M_\mathrm{exact}=X^\top X.
$$

无重复候选索引的 exact design 是大小 n 的子集 S；允许重复实验则应用非负整数次数 z_i，sum z_i=n。两者应使用**不同契约**，不能把“相同坐标”与“相同候选 ID”混同。

近似设计使用单纯形权重 w_i≥0, sum_i w_i=1：

$$
M(w)=\sum_{i=1}^{m} w_i f_i f_i^\top
    =F^\top\operatorname{diag}(w)F.
$$

实际张量化应采用 weighted GEMM 而不是显式构造 diag(w)。B 个实例的 F 形状 (B,m,p)、w 形状 (B,m)、M 形状 (B,p,p)；单实例分别为 (m,p)、(m)、(p,p)。

如果精确实验频率为 w_i=z_i/n，则 M_exact=n M(w)，因此在 M≻0 时：

$$
\log\det M_\mathrm{exact}=p\log n+\log\det M(w),\quad
\operatorname{tr}(M_\mathrm{exact}^{-1})=\frac{1}{n}\operatorname{tr}(M(w)^{-1}),\quad
\lambda_{\min}(M_\mathrm{exact})=n\lambda_{\min}(M(w)).
$$

**不能跨 n、模型尺度或归一化约定直接比较原始准则数值。**

建议保持以下研究层级：物理因素坐标 C → 特征/Jacobian F → 噪声模型与信息矩阵 M → 线性代数/预测方差内核 → D/A/E/G/I/C/V/S/T 评分 → exact/approximate/sequence 搜索 → 数值证书。组合 DOE/QMC 产生候选点但不自动构成 OED 最优解。

## 2. 多项式特征与模型秩

线性模型（含截距）在 k 个因素下 p=k+1：

$$
f(c)=[1,c_1,\ldots,c_k]^\top.
$$

完整二次响应面模型：

$$
f(c)=[1,\{c_j\},\{c_j^2\},\{c_i c_j:i<j\}]^\top,\qquad
p=1+k+\frac{k(k+1)}2.
$$

完整至 d 阶多项式的项数理论上 p=binom(k+d,d)，与所选基底及是否存在约束有关。PyDOE 固定版本的 build_design_matrix 显式生成截距、线性、纯平方、二元交互，degree≥3 时枚举组合重复的高阶项；TorchOED 应明确阶数、项顺序、幂次索引和特征缩放，不把物理坐标矩阵误作设计特征矩阵。

模型参数可识别性的必要条件是 n≥p 且 rank(X)=p。坐标上看似分散的设计未必能估计指定多项式模型。对混合实验尤其重要：若 q 个配比 x_j≥0 且 sum_j x_j=1，把截距 1 和全部 q 个线性配比一起放入模型将**必然秩亏**，因为 1=sum_j x_j。应使用 Scheffé 混合模型等适当参数化，而不是隐式加 ridge。

**独立 CPU_CHECK：** 4 行三组分混合数据的 [1,x1,x2,x3] 四列特征矩阵秩仅为 3；与理论一致。

## 3. 信息矩阵的更一般结构：Fisher、噪声、参数不确定性

对于在参数 θ 处计算的 Jacobian J(θ)∈R^{n×p}、观测噪声协方差 Σ≻0，局部 Fisher 信息：

$$
M(\theta)=J(\theta)^\top\Sigma^{-1}J(\theta).
$$

独立异方差时 Σ=diag(σ_i²)，可化为加权外积 sum_i σ_i^{-2} j_i j_i^T；完整相关 Σ 包含观测之间的交叉项，**不是**单纯的逐行标量 weighted GEMM，也不天然满足普通一行增删时的秩一更新结构。可通过 Cholesky whitening 或一般线性求解，但选中行集改变时须正确截取 Σ_SS 并重新考虑复杂度。

PyDOE 固定源文件 optimal/fisher.py 的 fisher_information 输入 Jacobian 形状 (N,p) 或 (N,p,K)，后者先分别计算信息矩阵再对参数空间点求平均：

$$
\bar M=\frac1K\sum_{s=1}^K J(\theta_s)^\top\Sigma^{-1}J(\theta_s).
$$

**不同目标不可混淆：**

$$
\log\det\left(\frac1K\sum_s M_s\right)
\quad\ne\quad
\frac1K\sum_s\log\det M_s
$$

通常前者 ≥ 后者（logdet 凹性），甚至可能平均信息满秩而部分 θ 点的信息奇异。要明确是“平均 Fisher 信息”准则、Bayesian expected logdet 还是 worst-case robust design。

**独立 CPU_CHECK：** M1=diag(100,1), M2=diag(1,100)。平均信息的 logdet ≈7.84394667，而平均 logdet ≈4.60517019，差异显著。

固定 PyDOE 的 jacobian_optimal_design 允许重复选取候选行（replicates），且评分时内部加上 1e-10·I。**TorchOED 不应把这种隐式 regularization 直接继承为未正则化的 D/A/E/S 目标**。模型的相关性、重复索引与噪声协方差还需要明确可行性约定。

## 4. 最优性准则数学图谱：D/A/E/G/I/C/V/S/T

以下 M 默认对称正定，除特殊声明外采用**最小化风险/方差的统计含义**，并同时注明 PyDOE 固定版本“最大化 score”的符号约定。使用线性 solve/Cholesky 而不是生产热路径显式求逆；所有尺度因子与 σ² 必须在接口明示。

| 准则 | 统计目标/数学定义 | PyDOE 1.6.0 函数行为 | 共用张量内核及风险 |
|---|---|---|---|
| D | max logdet(M)，最小化参数置信椭球体积 | d_optimality 返回 det(M) | Cholesky/slogdet；logdet 单调等价但值不同 |
| A | min tr(M^{-1})，平均参数方差 | 返回 -tr(regularized_inv(M)) | 批量 solve、对角迹；PyDOE 可能隐式 ridge |
| E | max λ_min(M)，最弱信息方向 | 返回最小特征值 | eigvalsh；重根处非光滑 |
| G | min max_{z∈Z} f(z)^T M^{-1} f(z) | 返回负的候选集最大预测方差 | batch solve + quadratic form + max |
| I | min tr(H M^{-1})，积分/加权平均预测方差 | 返回 -tr(H·regularized_inv(M)) | moment H 明确概率测度与归一化 |
| C | min c^T M^{-1}c，目标线性对比方差 | 返回其负数 | solve(M,c)，c 是参数空间向量 |
| V | min (1/r)sum_{z∈Z_test} f(z)^T M^{-1}f(z) | 返回负的有限测试点平均预测方差 | 固定测试特征矩阵；可转化为 moment |
| S | 与设计列尺度、行列式及相关性混合的特定目标 | det(M)/prod_j sqrt(max(M_jj,1e-15)) | 非经典纯正交性度量，依赖尺度 |
| T | 模型判别准则，定义必须针对两种模型核准 | 源码为 d^T P_X d，P_X=X(X^T X)^{-1}X^T | **不等同于一般残差式 T 判别目标** |

### 4.1 I 与 V 的严格等价条件

若测试特征矩阵 Z∈R^{r×p}，测试点均匀加权，令 H_Z=Z^T Z/r，则：

$$
\frac1r\sum_{j=1}^r z_j^\top M^{-1}z_j
=\operatorname{tr}(H_Z M^{-1}).
$$

所以在测试特征集/概率权重一致时，**V 与离散 I 是同一个数学目标**。I 可通过积分测度 ν 定义 H=E_ν[f(x)f(x)^T]；V 通过特定有限测试点定义 H。若概率测度、测试点或权重不同，二者并不自动相等。

**独立 CPU_CHECK：** 用候选行 (1,-1),(1,0),(1,1)，权重 (0.3,0.3,0.4)，同一组评估点求得 I 与 V 均为 1.9806763285024156。

### 4.2 G 与 D 的关系：近似设计等价定理

对归一化近似设计 M(w)，d_i=f_i^T M(w)^{-1} f_i。D-optimal 的全局最优充要条件为：

$$
\max_i d_i=p
$$

前提：候选特征张成 R^p、允许连续单纯形权重、当前 M≻0。这是 Kiefer–Wolfowitz 等价定理；与最小化候选集最大预测方差的 G-optimality 密切联系。**不能直接推广成固定整数 n 和无重复精确设计的全局最优条件。**

### 4.3 S 准则的尺度敏感性

PyDOE 固定版本：

$$
S(M)=\frac{\det M}{\prod_j \sqrt{M_{jj}}}.
$$

设 M=tM0，维度 p，则 S(tM0)=t^{p/2}S(M0)。设相关矩阵 C=D^{-1/2}MD^{-1/2}, D=diag(M)，则 S(M)=sqrt(prod_j M_jj)·det(C)。因此它不仅奖励列正交性，也受到列尺度影响，不能当作尺度不变的纯正交指标。

### 4.4 T 准则的实质语义差异

PyDOE 固定版本计算 P_X=X(X^T X)^{-1}X^T 后返回 d^T P_X d，并对默认 model_diff_subset 使用零向量。因此若不提供模型差异，T score 为零而不能驱动判别优化。

对于经典“用竞争模型 X 拟合真实模型均值差异 d”所要求的残差判别量，一个自然目标却是：

$$
d^\top(I-P_X)d,
$$

而不是 d^T P_X d。后者测量**被 X 解释**的能量，前者测量**不能被 X 解释**的能量。两个方向不可擅自互换，必须核对具体经典 T-optimality 问题的零假设/竞争模型与投影结构。

**独立 CPU_CHECK 反例：** X=I_2, d=(2,-1)。PyDOE 的投影能量为 5，残差能量为 0。因为 d 完全落在 span(X)，按残差判别应没有可区分信号；因此“名字同为 T”不足以视为数学等价。本文不据此断言上游软件在其自定义契约下出错。

## 5. D/A/I/C 权重导数与共用敏感度原语

设 M(w)=sum_i w_i f_i f_i^T，w 在单纯形内且 M≻0，H≽0，c∈R^p。

对 **最大化形式** 目标：

$$
\Phi_D=\log\det M,\quad
\Phi_A=-\operatorname{tr}(M^{-1}),\quad
\Phi_I=-\operatorname{tr}(HM^{-1}),\quad
\Phi_C=-c^\top M^{-1}c,
$$

逐坐标导数为：

$$
\frac{\partial\Phi_D}{\partial w_i}=f_i^\top M^{-1}f_i,
$$

$$
\frac{\partial\Phi_A}{\partial w_i}=f_i^\top M^{-2}f_i=\|M^{-1}f_i\|_2^2,
$$

$$
\frac{\partial\Phi_I}{\partial w_i}=f_i^\top M^{-1}HM^{-1}f_i,
$$

$$
\frac{\partial\Phi_C}{\partial w_i}=(f_i^\top M^{-1}c)^2.
$$

推导使用 d(M^{-1})=-M^{-1}(dM)M^{-1} 及 dM/dw_i=f_if_i^T。由方向导数相对于 w→e_i，存在相应**近似设计**全局最优性条件（目标凹且定义域正定）：

- D：对所有 i 有 d_i^D≤p；有正权重的点达到等号。
- A：d_i^A≤tr(M^{-1})；正权重点达到等号。
- I：d_i^I≤tr(HM^{-1})；正权重点达到等号。
- C：d_i^C≤c^T M^{-1}c；正权重点达到等号。

注意 c-optimal 的奇异设计可能仍有可估计的特定对比；这里为了统一 Cholesky 契约限定 M≻0，不能把这个限制误称为 c-optimality 的一般必要条件。

对于 E=max λ_min(M)，最小特征值简单时 dλ_min/dw_i=(v_min^T f_i)^2；若重根则不可假定有唯一梯度。对于 G=-max_j f_j^T M^{-1}f_j，max 的活跃候选集合变化会导致非光滑性，必须基于活跃集做次梯度/方向导数研究。S 与 T 更不能盲用上述四个同一梯度公式。

**本次独立 CPU_CHECK：** Python 环境 PyTorch 2.10.0+cpu、float64；F=[(1,-1),(1,0),(1,1)]，w=(0.3,0.3,0.4)，H=F^T F/3，c=(0,1)。D/A/I/C 的 PyTorch autograd 梯度与以上解析式逐项一致；四种最大绝对误差分别 0、4.44e-16、0、4.44e-16。CUDA、TorchOED 工程测试均 NOT_RUN。

## 6. 上游同名 API 的数学契约差异清单

审计固定 commit 的 [criterion.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/criterion.py)、[utils.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/utils.py)、[efficiency.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/efficiency.py)、[fisher.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/fisher.py)、[optimal.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/optimal.py)。

| 事项 | 固定上游行为 | TorchOED 科学建议 |
|---|---|---|
| information_matrix | 默认 normalized=True，使用 X^TX/n，支持 alpha/X0 augmentation | M_exact=X^TX 为独立定义；归一化必须显式 |
| D | det(M) 而非 logdet | 数值路径 prefer logdet；oracle 比较需转换 |
| A/I/C/G/V | 多处调用 regularized_inv，可能在求逆警告/失败时加 1e-8·I | 禁止未声明 ridge；报告病态/奇异 |
| d_efficiency | 100·det(X^TX/n)^(1/p)，是**绝对尺度指标** | 与相对最优效率 (det M/det M*)^(1/p) **分开命名** |
| a_efficiency | 100·p/tr((X^TX/n)^-1)，同样是尺度指标 | 不能不核实就当相对 A-efficiency |
| I | moment_matrix，候选点经验分布可形成 H | 显式评价测度 H 与积分误差 |
| V | test_points，平均测试预测方差 | 可约化为 H，但测试分布必须冻结 |
| S | det(M)/prod sqrt(M_jj) | 单独说明列尺度敏感 |
| T | d^T P_X d，默认 d=0 | 与经典残差判别目标核对，不自动继承 |
| Jacobian OED | 允许重复、评分内部加 1e-10 ridge | 明确复制次数与数值/统计模型 |
| optimal_design | 签名默认 method="sequential"，docstring 却出现 default detmax 的描述 | 以实际签名与运行行为为准 |

科学证据不是“PyDOE 不好”，而是不同软件/文献可能采用不同定义。**TorchOED 目标是可审计的统计与数值契约，而非逐比特或逐函数兼容。**

## 7. 组合实验设计：Factorial、别名、正交阵、筛选与分组

固定上游根包 API 核查来源：[pydoe/__init__.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/__init__.py)。本节属于类别级数学研究；未逐行审计该类别所有实现。

| 家族与上游代表 | 数学结构 | 值得重用的 Torch 内核 | 主要证明/风险 |
|---|---|---|---|
| fullfact, ff2n | 离散水平集合笛卡尔积；m=prod_j L_j | meshgrid / cartesian_prod / 整数索引 | 组合大小指数增长；顺序与编码 |
| fracfact, fracfact_by_res, fracfact_opt | 二水平设计列在 GF(2) 上生成，定义关系与 aliasing | 符号张量、整数异或/位运算、子集筛选 | resolution 是所有定义关系最短词长度，而非只看单个生成元 |
| fracfact_aliasing, alias_vector_indices | 交互列乘积、别名等价类 | 批量 sign product、bitsets | 交互不可识别；不得报告虚假独立系数 |
| pbdesign, gsd | Plackett–Burman 主效应筛选；通用混合水平 subset | 矩阵索引与组合构造 | 主效应正交不等于交互可区分 |
| fold, john_three_quarter_design | 通过符号翻转/半折叠缓解特定别名 | 行/列符号变换 | 哪些 alias 被解除须逐设计证明 |
| latin_square, graeco_latin_square, hyper_graeco_latin_square, block_full_factorial | 拉丁方/多因素阻断，控制干扰因素 | permutation、索引、正交表 | 组合存在性及列互相正交限制 |

GF(2) 形式举例：二水平设计列可编码为 s_g∈GF(2)^r 并由点积得到取值 (-1)^{x·s_g}；列的乘积对应编码向量异或。若若干列的编码异或为 0，则对应交互与截距混淆。此表示能使 alias 检查变成整数/二元算子，而不是重复数值矩阵相乘。

**明确边界：** 经典组合构造往往是索引与离散代数问题，GPU 不必然更快；GPU 更有价值的部分可能是成批验证正交性、别名、模型秩及大规模评分。

## 8. 响应面设计与混合实验设计

**响应面设计（Response Surface）：** bbdesign (Box–Behnken)、ccdesign (Central Composite)、Doehlert shell/simplex、star、union、repeat_center、block_ccdesign、small_composite_design。它们构造具有特定几何对称性、曲率辨识能力、旋转性或分组性质的实验点集合；重要判据是二次模型列秩、条件数、预测方差的空间分布、中心点重复和 alpha/star 半径。若存在约束因素，标准 CCD 星点可能离开可行域；不能把解析对称性等同于实际可行性。

**混合实验（Mixture）：** simplex_lattice_design、simplex_centroid_design、mixture_axial_design、extreme_vertices_design、mixture_process_design。可行域在 q-1 维单纯形 x_j≥0, sum x_j=1 上；simplex lattice 以整数分拆 r_1+…+r_q=h, x_j=r_j/h 构造点，理论点数为 binom(q+h-1,h)；simplex centroid 使用非空组分子集，共 2^q-1 个中心点。Constrained mixture 极点需处理额外线性不等式，不能用普通笛卡尔网格替代。

**数学共性：** 这两大类更多复用 feature-map、笛卡尔/组合索引、几何约束与秩验证，而不是直接复用 D-opt 的全部交换算法。设计构造器输出物理候选点；再经特征映射与 OED 准则判断其优劣。

## 9. LHS、空间填充、QMC：与 OED 不同的几何目标

固定 PyDOE 类别包含 lhs、oa_lhd、sliced_lhs、nested_lhs、maximin_design、minimax_design、maxpro_design、nearly_orthogonal_lhs、random_uniform、random_k_means，以及 sobol_sequence、halton_sequence、hammersley_sequence、rank1_lattice、korobov_sequence、faure_sequence、niederreiter_sequence、sukharev_grid、cranley_patterson_shift。

### 9.1 LHS 及其结构

Latin Hypercube Sampling 要求每个坐标维度在 n 个一维分层中各出现一次（边界点/随机抖动有多种实现），实现为每列 independent random permutation + strata jitter。**每维均匀分层不保证多维联合空间填充、两维正交或模型参数可识别。**

OA-LHD 在正交阵基础上改进二维分层；sliced LHS 要求每个 slice 也保留边际分层；nested LHS 的小设计是大设计的特定子集（与多保真成本结构相关）。它们的核心是组合 permutation 与嵌套索引约束，不能只用一个通用随机矩阵实现。

### 9.2 空间填充的三种不同目标

给定 X={x_i}：

- **Maximin:** max_X min_{i<j} ||x_i-x_j||，优先点之间的最小间距。
- **Minimax covering:** min_X max_{z∈Ω} min_i ||z-x_i||，优先域中最远位置被覆盖的距离；有限候选网格替换 Ω 会引入离散近似。
- **MaxPro:** 典型目标为 min Σ_{i<j} prod_{d=1}^k (x_{id}-x_{jd})^{-2}，偏向所有投影的填充；需要处理相等坐标带来的奇异/极值并采用稳健的对数计算。

另有 nearly-orthogonal LHS 的 max-column-correlation 目标，不能与 G-optimality 的 worst prediction variance 混淆。批量 pairwise distance、tiled cdist、相关矩阵及 logsumexp/prod 是 PyTorch 候选原语。距离矩阵占用 O(n²) 内存；大规模必须考虑 tiling。

### 9.3 QMC 与 Sobol 复用

Sobol/数字网、Halton/Hammersley、rank-one/Korobov lattice、Faure/Niederreiter 等属于低差异点集与积分/采样理论，不是“优化信息矩阵行列式”。主要研究 discrepancy、数字构造、样本数限制、scrambling 和重现性。优先复用 torch.quasirandom.SobolEngine；需核查具体 Sobol 序列参数、设备与 dtype 支持，不能假定结果与 PyDOE 完全一致。Cranley–Patterson shift 为一个随机平移模 1 的随机化步骤，必须清晰区分确定性与随机化保证。

## 10. Sensitivity、Sparse Grid、Taguchi 与序贯设计

| 家族 | 数学对象 | 研究重点 |
|---|---|---|
| Morris OAT | 输入轨迹上的有限差分 elementary effects | 单变量扰动步长、轨迹与随机种子；适合批量函数评估 |
| Saltelli/Sobol sensitivity | 基于方差分解的一阶/总效应指标 | 输入独立性与采样协定；高维批量生成/函数评估 |
| Iman–Conover | 排序变换实现目标秩相关结构 | 保持边际分布，不保证精确 Pearson 相关矩阵 |
| Sparse grids / Smolyak | 多层一维求积/插值张量积的稀疏组合 | 组合系数、节点重复、维度灾难的缓解而非消除 |
| Taguchi OA + SNR | 正交表、稳健设计、目标相关信噪比 | SNR 的 larger/smaller/nominal 最优定义不可混用 |
| Definitive screening | 特定三水平筛选矩阵与效应别名结构 | 样本量、主效应/交互与二次项的可识别性 |
| Supersaturated design | 参数/因素多于试验次数，依赖稀疏效应假设 | 不能按普通满秩 OLS/Fisher inverse 直接解释 |
| Gaussian-process sequential design | 后验均值/方差 + acquisition | 与线性模型 D-opt 的统计假设不同 |

常用最大化设定下，Bayesian Optimization 的采集函数包括：
- Expected Improvement（EI）：E[max(f(x)-f_best-ξ,0)]；
- Probability of Improvement（PI）：Pr(f(x)>f_best+ξ)；
- Upper Confidence Bound（UCB）：μ(x)+κσ(x)。

对正态预测分布，可使用 Φ、φ 的闭式形式（σ→0 的极限必须单独处理）。这些属于 GP posterior + acquisition + sequential update 的新数学层，不能仅依靠信息矩阵 GEMM 就宣称完成。

**整体判断：** PyDOE 是多种数学领域的集合，远非“91 种 OED 矩阵优化算法”。按共享原语拆分后，哪些算法可高效上 GPU、哪些主要依赖小规模离散组合，将清晰得多。

## 11. Fedorov：单点交换精确公式与秩条件（继承此前所有导师研究）

考虑选中设计 A∈R^{n×p}、M=A^TA≻0。交换选中行 a 与新候选行 b：

$$
M'=M-aa^\top+bb^\top.
$$

令 U=[-a,b], V=[a,b]，通过矩阵行列式引理：

$$
\frac{\det M'}{\det M}
=\det(I_2+V^\top M^{-1}U)
=(1-u)(1+v)+c^2=:\Gamma,
$$

其中 u=a^TM^{-1}a、v=b^TM^{-1}b、c=a^TM^{-1}b。只有 M'≻0 才可以使用 log Γ 表示有限的 logdet 改善；Γ=0 表示交换后秩亏。固定 n 的 D-opt 换入改善条件为 Γ>1。

选中行杠杆分数 h_i=a_i^TM^{-1}a_i 是正交投影 P=A(A^TA)^{-1}A^T 的对角元：

$$
0\le h_i\le1,\qquad \sum_i h_i=p,\qquad
h_i=1\iff \operatorname{rank}(A_{-i})<p.
$$

特别是 n=p 时所有 h_i=1：**删除中间态必定降秩，但完整交换仍可合法**。例如 A=[(1,-1),(1,1)]，用 b=(1,0) 替换 a=(1,-1) 后，原始 det M=4，新 det M'=1，比率 Γ=1/4。此时 u=1，v=1/2，c=1/2；不能依赖删除后矩阵求逆。

满秩初始精确设计存在的充要条件（独立索引、无重复、n≤m）：n≥p 且 rank(F)=p。可采用列主元 QR 思想在 F^T 上选线性独立行：给定当前正交基 Q_k，候选残差 r_j=(I-Q_k Q_k^T)f_j；选残差范数较大的方向，而不是只选原始范数大的点。数值秩需要经验证的阈值，不应自称精确代数秩。

**初始化反例：** F=[(1,0),(2,0),(0,1)]，n=p=2。按范数贪心与低索引 tie-break 可能选出平行的 (2,0),(1,0)，det=0；选择 (2,0),(0,1) 则 det=4。

## 12. 单点交换的张量化与数值鲁棒性

令 M=LL^T，R=L^{-1}A^T∈R^{p×n}，T=L^{-1}B^T∈R^{p×m_new}，均通过三角线性求解而非显式逆矩阵得到。

$$
u_i=\sum_{\ell=1}^p R_{\ell i}^2,\qquad
v_j=\sum_{\ell=1}^p T_{\ell j}^2,\qquad
C=R^\top T,\qquad
\Gamma_{ij}=(1-u_i)(1+v_j)+C_{ij}^2.
$$

可以 tile 计算 m_new 候选，避免无条件分配 n×m_new 的显存。一次迭代算术量预估 O(p³+(n+m_new)p²+n m_new p)，不包含所有控制/候选生成/校验开销。比逐个交换重新构造 Gram 并分解具有共享运算优势，但真实端到端 CUDA 加速**没有测量证据**；小规模可能被 kernel launch、同步及分块开销主导。

数值扰动的精确展开：

$$
\widetilde\Gamma-\Gamma
=-(1+v)\delta u+(1-u)\delta v-\delta u\delta v
  +2c\delta c+(\delta c)^2
  +\delta_{\rm arithmetic},
$$

因此具备可信 E_u,E_v,E_c 与最终舍入界时：

$$
|\widetilde\Gamma-\Gamma|
\le (1+v)E_u+|1-u|E_v+E_uE_v+2|c|E_c+E_c^2+E_{\rm arithmetic}.
$$

**不能随便把 float32 machine epsilon 当作这个严格界。** 需分解误差、三角求解误差、GEMM 误差与条件数分析，尚 RESEARCH_PENDING。

若能证明 Γ∈[Γtilde-E, Γtilde+E]，则 lower>1 才是已证实改善，upper≤1 才是已证实无改善。所有合法交换上界≤1 才能宣称严格 1-swap 局部最优。固定 top-K 复核一般不能保证“真实最佳交换”或完整邻域局部最优。

对不确定候选，可通过完整候选设计 A'=QR 直接核验 2Σ log|R_ii|，从而避免形成 A'^T A' 后平方条件数；仍需警惕 QR 对极端病态的精度极限、设备行为和可信 rank 判断。不得静默裁剪负 Γ 或加入 ridge。

**此前独立 CPU_CHECK（PyTorch 2.10.0+cpu, float64/float32）：**
- seed 42073，A:(8,4)，B:(17,4)，136 次直接交换行列式比与共享 Cholesky 公式差异 max 8.88e-16（float64）。
- 近奇异案例 A=[(1,1.2),(1,1)]，将首行替换为 (1,1.0001)：理论原始 det=0.04，新 det=1e-8，Γ=2.5e-7；float64 公式约 2.499999991e-7，float32 公式却得到 -1.18045864e-6，计算杠杆值约 1.000000715>1；float32 Gram 行列式约 -4.7686e-7（错误符号），直接 QR logdet 约 -18.421234、理论约 -18.420681。
- 这说明严格数学上的 Γ≥0 在未经验证的 float32 算法里可能被数值误差破坏。并非 TorchOED 代码失败，软件尚未实现。

## 13. Fedorov 的局部陷阱、k-swap 与 DETMAX（继承并扩展）

定义单点合法交换邻域 N_1(S)。在精确计算下：

$$
S\text{ 是 1-swap 局部最优}\iff
\max_{T\in N_1(S)}\frac{\det M(T)}{\det M(S)}=1
$$

其中 N_1(S) 包含原 S。**这不是离散全局最优的证明。**

严格反例（m=4,n=p=2）：f1=(1,0), f2=(0,1), f3=(0.9,0.9), f4=(0.9,-0.9)。当前 S={1,2} 行列式=1，全部四种单点替换结果行列式均为 0.81；但 S*={3,4} 信息矩阵=1.62I，行列式=2.6244，logdet 增益≈0.9648523。此前及本次用 PyTorch CPU float64 全部 6 种子集穷举复核。单点严格局部最优可远离全局最优。

**PyDOE 当前实现的实际差异（固定 algorithms.py）：**
- fedorov：sequential_dykstra 初始化，每轮列举全部合法单点替换，接受最优严格改善。
- modified_fedorov：同样初始化；按已选位置顺序更新当前设计，一轮可进行多次连续单点替换，但候选池每轮冻结。**不是同时 2-swap**，无法直接跳过严格 1-swap 陷阱。
- detmax：先用 simple_exchange_wynn_mitchell；停滞后 best add one + best drop one。最终 (S∪{j})\\{i} 不是 S 就是普通 1-swap，故**当前源码的单层 add/drop excursion 并不能越过严格的 1-swap 局部最优**。经典 Mitchell DETMAX 允许多层 n±k excursion；不能只看名字就把当前函数当经典多层版本。

### 13.1 真正的 k 点交换矩阵引理

移除 k 行 O∈R^{k×p}，加入 k 行 N∈R^{k×p}，定义 W=[O^T,N^T]∈R^{p×2k}，J=diag(-I_k,+I_k)：

$$
M'=M+WJW^\top,\qquad
\frac{\det M'}{\det M}
=\det(I_{2k}+JW^\top M^{-1}W).
$$

即对 k 点交换只需计算一个 2k×2k 行列式，且不要求删除 k 行后的中间矩阵可逆。具体分块矩阵为：

$$
\begin{bmatrix}
I_k-OM^{-1}O^\top & -OM^{-1}N^\top\\
NM^{-1}O^\top & I_k+NM^{-1}N^\top
\end{bmatrix}.
$$

精确 k-swap 候选数是 binom(n,k)·binom(m-n,k)。对于不同候选索引的 exact design，任何大小为 n 的设计可通过最多 min(n,m-n) 个同时替换到达，因此完整枚举该最大邻域意味着全局搜索，但组合复杂度通常不可承担。

**此前 CPU_CHECK：** p=3,n=6,新候选 11,k=2，825 组双点替换的 4×4 行列式引理比率与直接 Gram 比率最大绝对误差≈2.22e-15（float64）。

**多层 excursion 几何示例：** 从 {1,2} 先加 3、再加 4，随后删 1、删 2，最终 {3,4}。不同大小中间设计的 det 依次 1→2.62→6.8644→4.2444→2.6244；**中间 n 不同，不能直接比较原始 determinant 来证明 fixed-n 更优**。这证明路径可达，并不证明经典 DETMAX 的任意贪心规则都会走该路径。

### 13.2 多起点不是扩大邻域

对四点反例的 6 个初始大小为 2 的设计，确定性 best-improvement 1-swap 从 5 个起点抵达 {3,4}，仅 {1,2} 停滞。若（仅在此例）每次独立且均匀地抽取这 6 个起点，R 次成功概率 1-(1/6)^R：R=1 83.33%，R=2 97.22%，R=3 99.54%，R=5 99.99%。对于偏置初始化或相关随机起点这些数值无保证。PyDOE 的默认 deterministic sequential_dykstra 不是随机重启。

## 14. 可验证质量保证：局部证书 + 全局松弛界

定义 N_k(S) 为最多 k 个索引同时交换的全部合法邻居且包含 S：

$$
\rho_k(S)=\max_{T\in N_k(S)}
\frac{\det M(T)}{\det M(S)}.
$$

若精确计算并完整评价 N_k，则 ρ_k(S)=1 等价于 k-swap 局部最优；只扫描被启发式剪枝后的子集则不满足完整证书要求。在四点反例 ρ_1=1、ρ_2=2.6244。

此外，有独立于任何交换算法的**全局质量上界**。对 n 次实验归一化 \bar M=M/n、候选分数 d_j=f_j^T \bar M^{-1} f_j、d_max=max_j d_j，任一同样实验次数的最优精确设计 S* 满足：

$$
\operatorname{tr}(\bar M^{-1}\bar M^\star)
=\frac1n\sum_{j\in S^\star}d_j\le d_{\max}.
$$

对 H=\bar M^{-1/2}\bar M^\star\bar M^{-1/2}，由正特征值 AM–GM：

$$
\frac{\det M^\star}{\det M}
=\det H\le\left(\frac{\operatorname{tr}H}{p}\right)^p
\le\left(\frac{d_{\max}}p\right)^p.
$$

因此：

$$
\log\det M^\star-\log\det M
\le p\log(d_{\max}/p),\qquad
\mathrm{D\!-\!Eff}_{\rm relative}(S)
=\left(\frac{\det M}{\det M^\star}\right)^{1/p}
\ge p/d_{\max}.
$$

本例 S={1,2}，\bar M=I/2、d_max=3.24、p=2，理论最优比率上界 (3.24/2)^2=2.6244，**恰好由 S* 达到**；相对 D-efficiency 实际与下界均为 2/3.24≈0.61728395。

**绝不能将这个 relative D-efficiency 与 PyDOE 函数 d_efficiency 返回的 100·det(\bar M)^{1/p} 混为一谈。** 后者不是最优性上界/下界证书。

可参考 Madan et al. (COLT 2019) 的特定条件下组合 D-optimal local search 近似比分析；不得未经验证将其常数结论套用于 PyDOE 当前代码、任意 n/p 或 GPU 浮点实现。

## 15. 研究优先级与可证伪实验矩阵（不授权工程实现）

| 研究层/未来优先级 | 值得独立验证的主张 | 最小科学 oracle | 当前状态 |
|---|---|---|---|
| M1 特征/构造 | 多项式阶数/列顺序/正交性/混合约束正确 | 解析小矩阵、秩、枚举组合 | 公式 RESEARCH_READY；软件 NOT_RUN |
| M2 信息矩阵 D/A/I/C | 梯度与归一化正确、无隐式 ridge | torch float64 autograd vs 解析；小型 SciPy optional oracle | 独立 CPU_CHECK PASS，工程 NOT_RUN |
| M2 E/G/S/T | 活跃集合、重复特征值、准则语义明确 | 特征值、投影反例、尺度不变量 | 数学研究完成部分；契约待冻结 |
| M3 Fedorov k=1 | 秩二评分与直接评分一致 | 随机满秩/饱和/接近秩亏 | CPU_CHECK PASS；CUDA NOT_RUN |
| M3 多点/DETMAX | 真正扩大邻域能越过严格局部陷阱 | 4 点严谨反例 + 825 组 2-swap | 数学+CPU_CHECK PASS；大规模未测 |
| M3 可靠接受 | 不会接受虚假的数值改善或谎称局部证书 | 接近奇异 float32 反例、QR/SVD 复核 | RESEARCH_PENDING，尚无严格通用误差界 |
| M4 GPU break-even | batched solve/GEMM/tiled reductions 优于 CPU | 固定 n,m,p, dtype, memory、含同步完整墙钟 | NOT_RUN，不可宣称加速 |
| Space filling | LHS 分层、MaxPro/Maximin 的真实约束/数值尺度 | permutation invariants、成对距离 tiny oracle | 类别研究，代码 NOT_RUN |
| Fisher/Sequential | correlated-noise FIM、模型不确定性语义稳定 | 小规模显式 Σ^{-1} 与 GP posterior | 类别研究，代码 NOT_RUN |

**GPU-first 并非 GPU-always：** 当问题是离散小型组合构造时，CPU 很可能足够；当存在大批量特征、候选评分、距离和线性求解时，优先验证原生 CUDA。禁止用 Python for-loop 内的反复小 kernel 调用冒充张量化性能；应记录 device/dtype、peak memory、setup、数据传输与同步。

**推荐的科学内核划分（架构假设）：**
1. Feature-map / constrained candidate generation；
2. Information + whitening (independent/heteroscedastic/correlated noise)；
3. Solve-based quadratic forms、moments、logdet/eigensystem；
4. Score kernels for criteria and batch exchanges；
5. Search policies (sequential, best-swap, modified, true excursions, restarts)；
6. Quality diagnostics (rank, local-swap certificate, global relaxation bound, numerical uncertainty)；
7. Independent small-oracle and native CPU/CUDA evidence.

这是一份研究路线与数学契约汇编，**不构成要求 Codex 在 M0 一次性实现上述七层的工作单**。M0 仍只负责上游审计、覆盖清单和最小项目 spine；后续算法必须按其自身里程碑独立批准和核验。

## 16. 来源、证据链与后续更新纪律

### 16.1 固定源码（优先于名称/二手概述）

- [PyDOE 1.6.0 — PyPI 发布信息](https://pypi.org/project/pydoe/1.6.0/)（BSD-3-Clause；2026-10-21）。
- [PyDOE root public exports](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/__init__.py)（91 __all__ 名，不是算法数）。
- [Optimal criterion implementation](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/criterion.py)（D/A/E/G/I/C/V/S/T）。
- [Information matrix & criterion routing](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/utils.py)。
- [D/A efficiency definitions](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/efficiency.py)。
- [Polynomial feature map / moment matrix](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/model.py)。
- [Fisher / Jacobian experimental design](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/fisher.py)。
- [Exact design search algorithms](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/algorithms.py)。
- [Top-level optimal_design facade](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/optimal/optimal.py)。
- [PyDOE algorithm catalogue](https://pydoe.github.io/pydoe/)；[randomized/space-filling reference](https://pydoe.github.io/pydoe/reference/randomized/)；[changelog](https://pydoe.github.io/pydoe/changelog/)。
- [SAS/QC OPTEX Search Methods](https://support.sas.com/documentation/cdl/en/qcug/68161/HTML/default/qcug_optex_sect071.htm)（经典 Modified Fedorov、DETMAX 多层 excursion 与上游实现对照）。
- [Madan et al., Combinatorial Algorithms for Optimal Design, COLT 2019](https://proceedings.mlr.press/v99/madan19a.html)（理论近似界的前提需要精读）。

### 16.2 之前 SAE Scientific Mentor 的研究记录

- [Issue #1：首次 R1–R3 信息矩阵/等价定理](https://github.com/JohnnyZPiggy/TorchOED/issues/1#issuecomment-6481898989)。
- [Issue #1：单点交换、杠杆、浮点反例](https://github.com/JohnnyZPiggy/TorchOED/issues/1#issuecomment-6481980380)。
- [Issue #1：满秩初始化、局部陷阱、误差区间与全局质量界](https://github.com/JohnnyZPiggy/TorchOED/issues/1#issuecomment-6482073884)。
- [Issue #1：k-swap、Modified Fedorov、DETMAX、随机重启](https://github.com/JohnnyZPiggy/TorchOED/issues/1#issuecomment-6482154558)。

### 16.3 更新规范

每轮导师研究更新本文件时：记录适用的固定上游 SHA、数学假设、可证伪实验、数据类型与设备、明确区分推导与实际 PASS/NOT_RUN；修正文献约定冲突；保持 README/DEVELOPMENT_PLAN 的阶段边界；避免未经授权新增 Engineer Issue、实现/提交代码、自动合并任何 PR。

**本次文档完成判定：** 多类别数学研究+既有 Fedorov 结果整理完成；真实 TorchOED 包、CUDA/GPU 性能与可证明浮点误差证书尚未完成。


---

## 17. 全算法数学审计状态：基于真实 91 个公开导出（2026-10-28 增补）

**结论先行：截至本次研究，绝没有完成对所有 91 个公开 API 的逐项严格数学推导、张量化等价证明和 CPU/CUDA 验收。** 本节是透明的 research coverage ledger，不是 TorchOED 已实现功能表，也不能代替 Issue #1 规划的 Engineer \`docs/ALGORITHM_COVERAGE.md\`。

数据来源：固定 [PyDOE 1.6.0 root exports](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/__init__.py) 与 [完整冻结 Git tree](https://github.com/pydoe/pydoe/tree/866d3f411942cb662af8054562dc1f7e07372401/pydoe)。完整 tree 已获取，未截断，共 296 个条目（包含 pydoe 之外的内容与二进制/测试/其他文件）；不意味着每个 Python 源文件都经过逐行审读。

审计状态：
- **MATH_CPU：** 该名字对应的主要数学结构或其共享内核已有独立 PyTorch CPU 小例子验证；**不代表**其 PyDOE 所有 API 参数、随机序列或算法优化流程被逐项复现，更不是 TorchOED 工程 PASS。
- **SOURCE_MATH：** 已读相关固定源码中的具体函数/定义，并分析其结构/语义，但该公开函数的完整数值与形状契约未逐项独立验证。
- **FAMILY_ONLY：** 只有家族级数学分类或借鉴结论，尚不能声称单独审阅此导出函数的实现或建立其张量等价性。

**本轮分类结果：** \`MATH_CPU = 13\`、\`SOURCE_MATH = 32\`、\`FAMILY_ONLY = 46\`，合计 **91**。不是“已完成 91 个算法”。

| PyDOE root export | 所属公开模块 | 数学研究等级 |
|---|---|---|
| `alias_vector_indices` | `factorial` | FAMILY_ONLY |
| `block_full_factorial` | `factorial` | FAMILY_ONLY |
| `ff2n` | `factorial` | SOURCE_MATH |
| `fold` | `factorial` | FAMILY_ONLY |
| `fracfact` | `factorial` | FAMILY_ONLY |
| `fracfact_aliasing` | `factorial` | MATH_CPU |
| `fracfact_by_res` | `factorial` | SOURCE_MATH |
| `fracfact_opt` | `factorial` | SOURCE_MATH |
| `fullfact` | `factorial` | SOURCE_MATH |
| `graeco_latin_square` | `factorial` | FAMILY_ONLY |
| `gsd` | `factorial` | FAMILY_ONLY |
| `hyper_graeco_latin_square` | `factorial` | FAMILY_ONLY |
| `john_three_quarter_design` | `factorial` | FAMILY_ONLY |
| `latin_square` | `factorial` | FAMILY_ONLY |
| `pbdesign` | `factorial` | FAMILY_ONLY |
| `extreme_vertices_design` | `mixture` | FAMILY_ONLY |
| `mixture_axial_design` | `mixture` | FAMILY_ONLY |
| `mixture_process_design` | `mixture` | FAMILY_ONLY |
| `simplex_centroid_design` | `mixture` | SOURCE_MATH |
| `simplex_lattice_design` | `mixture` | MATH_CPU |
| `a_efficiency` | `optimal` | SOURCE_MATH |
| `a_optimality` | `optimal` | MATH_CPU |
| `build_design_matrix` | `optimal` | SOURCE_MATH |
| `build_uniform_moment_matrix` | `optimal` | SOURCE_MATH |
| `c_optimality` | `optimal` | MATH_CPU |
| `criterion_value` | `optimal` | SOURCE_MATH |
| `d_efficiency` | `optimal` | SOURCE_MATH |
| `d_optimality` | `optimal` | MATH_CPU |
| `detmax` | `optimal` | SOURCE_MATH |
| `e_optimality` | `optimal` | SOURCE_MATH |
| `fedorov` | `optimal` | MATH_CPU |
| `fisher_information` | `optimal` | SOURCE_MATH |
| `g_optimality` | `optimal` | SOURCE_MATH |
| `generate_candidate_set` | `optimal` | SOURCE_MATH |
| `i_optimality` | `optimal` | MATH_CPU |
| `information_matrix` | `optimal` | SOURCE_MATH |
| `jacobian_optimal_design` | `optimal` | SOURCE_MATH |
| `modified_fedorov` | `optimal` | SOURCE_MATH |
| `optimal_design` | `optimal` | SOURCE_MATH |
| `s_optimality` | `optimal` | SOURCE_MATH |
| `sequential_dykstra` | `optimal` | SOURCE_MATH |
| `simple_exchange_wynn_mitchell` | `optimal` | SOURCE_MATH |
| `t_optimality` | `optimal` | SOURCE_MATH |
| `v_optimality` | `optimal` | MATH_CPU |
| `bbdesign` | `response_surface` | FAMILY_ONLY |
| `block_ccdesign` | `response_surface` | FAMILY_ONLY |
| `ccdesign` | `response_surface` | SOURCE_MATH |
| `doehlert_shell_design` | `response_surface` | FAMILY_ONLY |
| `doehlert_simplex_design` | `response_surface` | FAMILY_ONLY |
| `repeat_center` | `response_surface` | FAMILY_ONLY |
| `small_composite_design` | `response_surface` | FAMILY_ONLY |
| `star` | `response_surface` | FAMILY_ONLY |
| `union` | `response_surface` | FAMILY_ONLY |
| `GaussianProcessRegressor` | `sequential` | FAMILY_ONLY |
| `expected_improvement` | `sequential` | MATH_CPU |
| `probability_of_improvement` | `sequential` | SOURCE_MATH |
| `sequential_design` | `sequential` | FAMILY_ONLY |
| `upper_confidence_bound` | `sequential` | SOURCE_MATH |
| `cranley_patterson_shift` | `space_filling.quasi_random` | FAMILY_ONLY |
| `faure_sequence` | `space_filling.quasi_random` | FAMILY_ONLY |
| `halton_sequence` | `space_filling.quasi_random` | FAMILY_ONLY |
| `hammersley_sequence` | `space_filling.quasi_random` | FAMILY_ONLY |
| `korobov_sequence` | `space_filling.quasi_random` | FAMILY_ONLY |
| `niederreiter_sequence` | `space_filling.quasi_random` | FAMILY_ONLY |
| `rank1_lattice` | `space_filling.quasi_random` | FAMILY_ONLY |
| `sobol_sequence` | `space_filling.quasi_random` | SOURCE_MATH |
| `sukharev_grid` | `space_filling.quasi_random` | FAMILY_ONLY |
| `lhs` | `space_filling.stochastic` | MATH_CPU |
| `maximin_design` | `space_filling.stochastic` | MATH_CPU |
| `maxpro_design` | `space_filling.stochastic` | MATH_CPU |
| `minimax_design` | `space_filling.stochastic` | FAMILY_ONLY |
| `nearly_orthogonal_lhs` | `space_filling.stochastic` | FAMILY_ONLY |
| `nested_lhs` | `space_filling.stochastic` | FAMILY_ONLY |
| `oa_lhd` | `space_filling.stochastic` | SOURCE_MATH |
| `random_uniform` | `space_filling.stochastic` | FAMILY_ONLY |
| `sliced_lhs` | `space_filling.stochastic` | FAMILY_ONLY |
| `TaguchiObjective` | `taguchi` | FAMILY_ONLY |
| `compute_snr` | `taguchi` | FAMILY_ONLY |
| `get_orthogonal_array` | `taguchi` | SOURCE_MATH |
| `list_orthogonal_arrays` | `taguchi` | FAMILY_ONLY |
| `taguchi_design` | `taguchi` | FAMILY_ONLY |
| `random_k_means` | `clustering` | FAMILY_ONLY |
| `morris_sampling` | `sensitivity_analysis` | SOURCE_MATH |
| `saltelli_sampling` | `sensitivity_analysis` | MATH_CPU |
| `doe_sparse_grid` | `sparse_grid` | SOURCE_MATH |
| `sparse_grid_dimension` | `sparse_grid` | SOURCE_MATH |
| `definitive_screening_design` | `specialized` | FAMILY_ONLY |
| `supersaturated_design` | `specialized` | FAMILY_ONLY |
| `iman_conover` | `utils` | FAMILY_ONLY |
| `scale_samples` | `utils` | FAMILY_ONLY |
| `var_regression_matrix` | `utils` | FAMILY_ONLY |

### 17.1 本轮深入的数学结构 A：Latin Hypercube / MaxPro / Maximin 的局部评分更新

PyDOE [maxpro.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/space_filling/stochastic/maxpro.py) 和 [distance_designs.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/space_filling/stochastic/distance_designs.py) 都从按列排列的 LHS cell center 初始化，然后交换一个坐标列的两行，逐轮从头计算完整评分。设设计矩阵 X∈R^{n×d}；每个固定列 j 的 cell index 是 0,…,n-1 的一个排列；交换同一列两行 r,s **严格保留 Latin 分层约束**。

MaxPro 评分设：

$$
\psi(X)=\sum_{1\le i<t\le n}g(x_i,x_t),\qquad
g(x_i,x_t)=\prod_{\ell=1}^{d}(x_{i\ell}-x_{t\ell})^{-2}.
$$

因为只交换 r,s 两行的一个因子列，所有其他点对 (i,t) 的贡献不变；此外点对 (r,s) 在该列只改变差的符号，平方后**其贡献也不变**。于是（点对所有坐标不同的 LHS cell-center 场景）精确增量为：

$$
\Delta\psi=
\sum_{t\notin\{r,s\}}
[g(x'_r,x_t)+g(x'_s,x_t)-g(x_r,x_t)-g(x_s,x_t)].
$$

这把每次交换重新执行 O(n²d) 点对评分变成 O(nd) 的受影响边计算；若已有完整成对分数缓存，还可进一步研究更新代价，但缓存 O(n²) 内存。**这是单步分数更新的数学计算量，不自动等于整个搜索复杂度或 GPU 必然提速。** LHS cell centers 保证同一列互不相同，避免零分母；其他候选样本可能存在重复坐标，必须额外约束。

Maximin 目标为 \(h(X)=\min_{i<t}\|x_i-x_t\|_2\)。缓存未受影响的点对最小值 h_rest 后，一次交换只需：

$$
h(X')=\min\bigl(
h_\mathrm{rest},\min_{t\ne r,s}\|x'_r-x_t\|_2,
\min_{t\ne r,s}\|x'_s-x_t\|_2,
\|x'_r-x'_s\|_2
\bigr).
$$

**本轮独立 torch float64 CPU_CHECK**：seed=271828，72 种 n=6…10、d=2…5 的真实两行/一列交换；MaxPro 完整重算与受影响点对公式的最大 scaled absolute error 2.274e-13，Maximin cached remainder 与完整重算最大 absolute error 为 0，72 组均保留 LHS 的每列完整分层。这是数学 oracle，不是 PyDOE 本身安装运行或 TorchOED 工程比较。

**可证伪 GPU 命题：** 批量交换候选可以利用矢量化 edge contribution 和 scatter/gather；但许多重复小规模随机交换的控制与缓存更新仍可能令 CPU 较快。需要独立比较完整重算、torch CPU incremental 和 torch CUDA batched incremental 的同步端到端时间以及显存；未测。

### 17.2 结构 B：Fractional Factorial 的 GF(2) 别名

读取 [factorial/factorial.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/factorial/factorial.py) 的 \`fracfact_by_res\`、\`fracfact_aliasing\` 定义：对于二水平 ±1 列，列乘积可以等价表示为整数比特串 XOR。若 D=A·B，则交互作用 A:B 与因子 D 完全混淆。其本质是 \(\mathrm{GF}(2)\) 线性代数与子集 XOR，并不是浮点矩阵乘法求解的加速问题。

独立 torch CPU_CHECK：用 2³=8 个完全因素组合构造 A,B,C 及 D=A·B，验证 A·B 与 D 的 8 个取值全部相同。这一表示还允许快速验证定义关系/别名等价类，但以所有交互子集完整搜索将遭遇指数级组合复杂度；GPU 的 GEMM 并不能取消这一本质障碍。

### 17.3 结构 C：Mixture 的组合数、单纯形秩约束

固定 [mixture/simplex.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/mixture/simplex.py)：\`simplex_lattice_design(q,h)\` 输出全部非负整数向量 r∈Z_{\ge0}^q 且 Σ r_i=h，实际比例 x=r/h；点数精确为 \(\binom{q+h-1}{h}\)。\`simplex_centroid_design(q)\` 输出所有非空组分子集的重心，点数 \(2^q-1\)。

独立 torch CPU_CHECK：q=4,h=3 的所有多重组合生成 20×4 候选，各行总和精确为 1；在其中加入截距列后，5 列的数值秩仅为 4。**这是模型参数化中的严格线性依赖，不应通过隐式 ridge“修复”。**

张量/并行机会：枚举整数 composition 索引、归一化广播、按子集 popcount 构造 centroid；生成的候选数量依旧组合/指数增长，必须有大小上限。

### 17.4 结构 D：Saltelli/Morris 与 EI 属于不同类型的张量计算

读取 [sensitivity_analysis/saltelli.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/sensitivity_analysis/saltelli.py)：从联合 Sobol 基础行 [A_i,B_i]∈R^{2d} 生成 A、B 以及每个坐标 j 的杂交行 A_B(j)、可选 B_A(j)；对固定 d，输出形状分别为 N(d+2)×d 或 N(2d+2)×d。它是具有明确排列次序的复制+单坐标替换算子：输入基础采样与样本装配应分开；不能声称 torch Sobol 与 SciPy Sobol 随机序列逐点相同。

独立 torch CPU_CHECK：torch.quasirandom.SobolEngine 为 d=3,N=8 生成 8×6 基础点，装配 A、3 组单列替换、B，得到张量形状 8×5×3（可展平成 40×3）；对每个杂交行，除被替换的那个坐标外均严格保留 A 的值。两套基础点对应坐标偶然相等时，杂交行也可能整体与 A 一样，故“恰好有一个数值不同”不是普遍不变量。

读取 [sensitivity_analysis/morris.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/sensitivity_analysis/morris.py)：一次轨迹由基点、方向对角矩阵、下三角矩阵与随机置换矩阵构造；批量轨迹可张量化矩阵乘法，但必须验证连续相邻行只有一个因素变化、步长 ±Δ，避免只验证整体输出形状。

读取 [sequential/acquisition.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/sequential/acquisition.py)：在高斯预测 μ,σ 与 best 值 t 下，EI=(μ−t)Φ(z)+σφ(z)，z=(μ−t)/σ；σ=0 的分支取 max(μ−t,0)（最大化情形）。独立 torch float64 CPU_CHECK：μ=(-0.5,0,0.5,1)，σ=(0,0.3,0.7,1.2)，与 SciPy 正态密度的独立数值积分最大绝对差 2.22e-16。**EI 评估是逐点广播的 elementwise kernel，GP posterior fit/update 才有独立的矩阵求解难题**。不要把 acquisition score 的简单 GPU 并行误称为完整 Bayesian Optimization GPU 加速。

### 17.5 关键源码风险：当前 PyDOE Sparse Grid 不可直接作为严格 Smolyak Oracle

这是比想象中的 GPU 性能更重要的事实核验。[固定 \`sparse_grid/sparse_grid.py\`](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/sparse_grid/sparse_grid.py) 的函数文档称 Smolyak construction，但实际 \`_generate_sparse_grid_points(level, dimension, grid_type)\`：先产生中心、轴点和少量内部组合；去重后若不足目标点数，就从逐步展开的笛卡尔网格中**任取点补足给定数量**，而没有在该函数中使用 \`grid_type\` 来选择 Clenshaw–Curtis / Chebyshev / Gauss–Patterson 的一维嵌套节点和 Smolyak 多指数组合。即 **\`grid_type\` 在此核心生成函数中未被引用**。

\`_spdim_formula\` 中 n_level≥7 时使用相同的固定七次 d 多项式，不依赖 n_level。因此例如 d=2 时，level 7 和 level 8 的目标数量都由同一式给出 705；这不能作为多层真实 Smolyak 网格递增结构的通用正确性证据。以上是**源码直接可验证性质，不是已实际安装 PyDOE 并做输出对照**。

**Scientist/Architect 研究建议：** 将 PyDOE 此模块标记为 \`UPSTREAM_ORACLE_UNSUITABLE_FOR_STRICT_SMOLYAK\`；TorchOED 若在后续阶段实现真正的 sparse grid / Smolyak，必须独立指定层级节点族、差分算子 Δ_l、组合多指标集合、嵌套去重与可证明 quadrature exactness。文档此前第 10 节的“Smolyak”的用语仅是数学类别描述，**不能据此宣称上游实际实现符合严格 Smolyak 数学契约**。

### 17.6 其他已核查但尚未逐函数证明的源差异

- [distance_designs.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/space_filling/stochastic/distance_designs.py) 的 minimax 选择覆盖对象是**随机生成的有限候选集**；不是对连续超立方体真 sup 的精确解。未来可用已有候选集合上的 tiled cdist 或最近距离缓存加速，但需单独分析域离散化误差。
- [sobol.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/space_filling/quasi_random/sobol.py) 默认 \`use_pow_of_2=True\` 时，会把非 2 的幂次样本量向上取整；这与“输出恰好 n 行”的固定接口契约不同。Tensor reimplementation 不应静默改变样本数。
- [oa_lhd.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/space_filling/stochastic/oa_lhd.py) 利用 OA 等级分块的随机 permutation；保留一维 LHS 不等于自动拥有指定强度的全部二维/高阶 OA 性质，需对输入 OA 的强度与输出分组不变量分别检查。
- [response_surface/central_composite.py](https://github.com/pydoe/pydoe/blob/866d3f411942cb662af8054562dc1f7e07372401/pydoe/response_surface/central_composite.py) 提供 circumscribed、inscribed、face-centered 选项；这些是不同几何/可行性约束，不能使用单一无限制张量网格替代。
- 除上述显式源码，本节 FAMILY_ONLY 的其余导出项仍缺少函数级源码核查与数值可证伪验收。**下轮研究应优先这些空白，而不是重复 D-opt/Fedorov 已经充分覆盖的推导。**

### 17.7 下一阶段的分组导师研究顺序（不是工程实现指令）

1. **FAMILY_ONLY 中的 Response Surface、Taguchi、Latin Square / Fractional Factorial 构造：** 每个公开函数给出索引/秩/别名/几何/合法性不变量及小规模独立 CPU Oracle，判断纯 CPU integer operations 是否优于 GPU；
2. **Sparse grid 严格 Smolyak 真正数学定义及其与当前上游非等价性：** 设计一维嵌套规则与多指标组合的数学 oracle；
3. **LHS 的 sliced/nested/OA 与 Space-filling 细分指标：** 从交换局部评分推进到不同切片/嵌套约束的合法邻域；
4. **Fisher/correlated-noise/GP/sensitivity 复杂结构：** 区分纯张量式与 sequential updating，需要不同的共享内核；
5. 等上述数学研究就绪，再由 Architect 按里程碑授权 Engineer。**本导师回合不推进 M0 代码、不建立新 Issue/PR，不声称 CUDA 测量。**

