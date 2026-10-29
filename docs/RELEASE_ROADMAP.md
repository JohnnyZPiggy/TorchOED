# TorchOED — Release roadmap, scope gates and maturity definition

> SAE V3.3 · Chief Architect / Scientific Mentor · 2026-10-29 · PLANNING ONLY / NOT ENGINEER DISPATCH. The duration ranges below are estimates, not promised timelines. Read with [Development Plan](DEVELOPMENT_PLAN.md), [M1/M2 Numerical Contract](M1_M2_NUMERICAL_CONTRACT.md), and [Architecture Survey](ARCHITECT_ECOSYSTEM_SURVEY.md).

## 1. Ground-truth baseline

- M0 Issue #1: **merged** PR #4 at merge commit `5d72176354a00ee51aff9ea9fee6893f747a1cb6`, accepted head `ef90fa60199ccee0317c072c11371d869f0f2194`: installable torch-only package skeleton, 91 frozen PyDOE 1.6.0 root export names categorized, private analytic Gram CPU smoke, SAE V3.3 project entrypoints. No public DOE/OED algorithm implemented at M0.
- E0 Issue #2 / Draft PR #3: real-wheel CPU parity science accepted at head `1d59ae5da6c694230aba2e95f217b517cea692c8`; **still open and unmerged** when roadmap drafted. Its experiment report is separate from M0 implementation; no M1 or GPU authorization is inherited.
- Canonical M1 constructor math and CPU fixtures are in [§9 of M1/M2 contract](M1_M2_NUMERICAL_CONTRACT.md#9-m1-constructor-architecture-and-independent-cpu-fixtures-2026-10-29); M2 criteria/Fisher in §§2–6; M3 source and exchange oracles in Scientific Mentor document. No new Engineer issue or CLAIM is created by this roadmap.

## 2. Product focus and deliberate non-goals

**Value proposition:** PyTorch-native DOE/OED building blocks with explicit mathematical definitions, exact-index selection, small independent CPU oracles, tensorized batches, and evidence-led performance claims. Not a drop-in clone of all 91 PyDOE exports; many are helpers/classes and specialist algorithms. Do not import PyDOE, NumPy or SciPy as core numerical runtime backends. Do not add custom CUDA kernels or a second solver ecosystem until reproducible workload evidence warrants them.

**Mature** means someone outside the author team can install a tagged release, reproduce the examples, rely on documented shape/dtype/device/error semantics, inspect test/build provenance, and receive predictable fixes. Stars, README length, API count and single-kernel GEMM speedup are not maturity metrics.

## 3. Milestones, dependencies, falsifiable gates

| Milestone | Minimum implementation | Hard exit gate | Indicative effort after authorization |
|---|---|---|---|
| M0 — merged | packaging, 91-name status ledger, private Gram CPU oracle, pinned SAE | already accepted and merged; retain evidence | done |
| M1 — constructors | `full_factorial`, `ff2n`, random `lhs`, native-engine `sobol`, linear/quadratic feature map | §9 F/L/S/P CPU fixtures, seed isolation, exact row/term order, invalid-input/memory cap, batch/autograd as specifically scoped; no fake PyDOE bitwise parity | 1–2 weeks |
| M2 — criteria | raw/normalized/weighted information matrix, D/A/E on CPU float64, then C/G/I/V only with explicit contrast/evaluation measure | known analytical matrices, batch parity, rank-deficient and non-finite policies, no silent ridge, optional gradcheck on SPD inputs, no false efficiency % | 1–2 weeks |
| M3 — search | sequential greedy, exact-ID Fedorov, Modified Fedorov, DETMAX only after evidence | tiny exhaustive optima, legal selections/replicates, local-swap monotonicity and stopping semantics, independent seeded restarts, CPU speed/memory on frozen cases | 2–4 weeks |
| M4 — select breadth | targeted high-demand space filling/response surface/mixture constructors | independent invariant oracles for each added family, coverage ledger updates, source/license review | 2–6 weeks |
| M5 — robustness and release | packaging examples, version support policy, public reference API, reproducible benchmark CLI, docs/changelog, property/negative tests, maintenance and security path | external clean install incl wheel, Linux/Windows/macOS smoke where available, no hard-coded CPU, release artifacts reproducible, semantic-version policy and compatibility | 2–4 weeks partly parallel |
| M6 — acceleration | CPU batching/tiling, GPU only after explicit authorization and actual hardware measurements; potential cuOED separately gated | same math/ID semantics, end-to-end timed comparisons incl transfers/synchronization, peak memory, negative/failed runs, declared break-even | optional, 2–6+ weeks |

**Dependencies:** M0 merge → explicit Scientist M1 authorization → accepted M1 → accepted M2 → accepted M3; M4 and M5 can be selective/parallel once stable math API exists. E0 PR #3 merge is an independent Scientist decision, not an automatic prerequisite to start M1 if the necessary frozen oracle data are available and provenance remains pinned. Do not allow M0 Issue #1 open status to be misconstrued as unmerged implementation.

## 4. Release definitions (planning ranges, not commitments)

- **Alpha 0.1.0: ~2–4 weeks** of active scoped development after Scientist authorizes M1. Contains tested constructors + a useful subset of information/criteria. Explicitly experimental/unstable API. Must not claim complete OED optimizer.
- **Beta 0.5.0: ~6–10 weeks** cumulative. A practically useful OED solver with at least one validated exact-ID search and independent quality-gap oracle, reproducible CPU baseline, documentation and changelog. No blanket GPU acceleration claim.
- **Stable 1.0.0: ~3–6 months** cumulative with sustained engineering, review bandwidth, release hygiene, multiple environment tests, stable numerical/typing/error contracts, real users/examples and issue-triage practice. If 1.0 is defined as full PyDOE coverage or dedicated native CUDA/cuOED implementation, **6–12+ months** is plausible and cannot currently be evidenced.
- **Schedule risk factors:** Scientist review availability; real-wheel oracle mismatches; tensorization memory blow-up; numerical rank/conditioning; upstream behavioral drift; cross-platform packaging; user-facing backward compatibility; GPU break-even disappointment. Do not estimate throughput from count of AI-generated commits.

## 5. M1 next engineering handoff — PREPARED, NOT SENT

**Prerequisite:** live main must contain accepted M0 merge; fresh issue/PR conflict check; separate Scientist approval of the M1 engineering scope. A future bounded engineer task should implement only the five named M1 constructors/features under the §9 contract, tests F1–F4/L1–L4/S1–S5/P1–P6, and append exact environment and CPU run evidence. Create M1 issue/PR only if and when Scientist separately authorizes it and no duplicate exists. Existing E0 PR #3 immutable; no M2/M3 algorithms, CI Actions, GPU billing or paid services.

**Architect promotion gate:** check module exports and package metadata at exact tested and submitted SHAs, explicit factor/term order, CPU float64 analytic source-independence, LHS strata and seed reproducibility, Sobol non-power-of-two and skip semantics, rank/alias positive/negative cases, invalid inputs, torch-only runtime. A draft that imports SciPy for LHS or copies PyDOE behavior without naming the difference fails.

## 6. Practical mature-package checklist

1. User-facing contracts: named `dtype`/`device`/`generator`, normalized FIM, rank failure, candidate ID identity, and safe size caps.
2. Test pyramid: exact small analytical oracles, frozen real PyDOE dev-only comparisons, deterministic property tests, bounded CPU integration, optional authorized GPU parity and E2E timing.
3. Release engineering: clean wheel and sdist installs, pinned build provenance, license/NOTICE checks, release notes, changelog, examples, supported Python/torch matrix, installation troubleshooting.
4. Maintenance: issues/templates/contribution guide, ownership, deprecation policy, reproducible bug report instructions, responsible dependency upgrades.
5. Evidence-first marketing: publish speedups only for exact same end-to-end task & quality conditions. Prefer one genuinely well-tested Fedorov pathway over 50 shallow unsupported exports.

**SAE authority:** Architect may maintain this plan and review artifacts; Engineer implementation requires task CLAIM and authorized branch; Scientist alone approves merges and scope expansion. No feature implementation or GPU activity occurs as a side effect of this document.
