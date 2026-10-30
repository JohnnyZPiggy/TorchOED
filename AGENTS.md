# TorchOED — SAE V3.3 project entrypoint

SAE pin: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`. Configuration: [sae.yaml](sae.yaml).
Read the actual [pinned constitution](https://github.com/JohnnyZPiggy/SAE-Framework/blob/d84942e4890d9837ae0e6cbb379b85c3a95aba84/AGENTS.md), assigned role and task-relevant workflows via [.sae/README.md](.sae/README.md); do not silently upgrade or assume links were read.

## Scope and roles

- Scientist sets the objective and can redirect/stop work; Architect/Mentor owns specifications, mathematics and independent review.
- Engineer implements/tests on independent branches and submits evidence; **never self-approves or merges**, never writes protected main or force-pushes.
- M0 Issue #1 historically permitted only the installable torch-only spine, frozen upstream inventory, minimal SAE adoption and CPU smoke. Planning documents alone do not dispatch algorithms. M1 Issue #5 is merged. M2-A Issue #7 is merged. M3-A Issue #9 is merged. Historical Issue #11 authorized CPUfloat64 exact-ID no-repeat simultaneous k=2 scores and oracles, with no search loop; PR #12 has since merged. M3-B2-P0 Issue #13 was stopped by the Scientist. Modified Fedorov, DETMAX, and k-swap search remain unassigned.
- E0 Issue #2 / PR #3 is separate experiment work: do not change its files or branch while executing M3-B1.
- Preserve existing LICENSE and research/history. Read [DEVELOPMENT_PLAN](docs/DEVELOPMENT_PLAN.md), [Mentor research](docs/SCIENTIFIC_MENTOR_RESEARCH.md) and [M1/M2 architecture](docs/M1_M2_NUMERICAL_CONTRACT.md). The §25 rank15 correction supersedes the historical rank12 hypothesis for actual PyDOE.
- Production numerical dependency: torch only. PyDOE/NumPy/SciPy are development references, never numerical backends. Prefer native torch primitives; no custom CUDA.
- Reuse COLD/WARM evidence when source/runtime is unchanged; refresh relevant git/gh/Issue/PR state before work. Use EXPLORE → VALIDATE with exact source, real commands and PASS/FAIL/NOT_RUN/BLOCKED.
- Use the designated D: workspace, including Dropbox, for owned scratch/caches/results. Clean only task-owned disposable files and report residue.
- **Current state: NO ACTIVE ENGINEER DISPATCH.** Issue #11 (M3-B1) was completed and PR #12 merged; do not reclaim, rerun, or extend it. Issue #13 (M3-B2-P0) and its CPU/GPU probe were explicitly stopped by the Scientist; the existing branch is historical, not permission to continue. No GPU/Colab experiments, k=2 search, implementation, or automatic Engineer assignment without a new explicit Scientist instruction.
- Preserve existing historical Issue/PR evidence and branches. The latest Scientist stop overrides older task descriptions and open Issue status. Future authorized Engineer tasks require a fresh scoped instruction and independent review; Engineers never self-merge.
