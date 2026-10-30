# TorchOED — SAE V3.3 project entrypoint

SAE pin: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`. Configuration: [sae.yaml](sae.yaml).
Read the actual [pinned constitution](https://github.com/JohnnyZPiggy/SAE-Framework/blob/d84942e4890d9837ae0e6cbb379b85c3a95aba84/AGENTS.md), assigned role and task-relevant workflows via [.sae/README.md](.sae/README.md); do not silently upgrade or assume links were read.

## Scope and roles

- Scientist sets the objective and can redirect/stop work; Architect/Mentor owns specifications, mathematics and independent review.
- Engineer implements/tests on independent branches and submits evidence; **never self-approves or merges**, never writes protected main or force-pushes.
- M0 Issue #1 historically permitted only the installable torch-only spine, frozen upstream inventory, minimal SAE adoption and CPU smoke. Planning documents alone do not dispatch algorithms. M1 Issue #5 is merged. M2-A Issue #7 is merged. M3-A Issue #9 is merged. Historical Issue #11 authorized CPUfloat64 exact-ID no-repeat simultaneous k=2 scores and oracles, with no search loop; PR #12 has since merged. M3-B2-P0 Issue #13 was stopped by the Scientist. Modified Fedorov, DETMAX, and k-swap search remain unassigned.
- E0 Issue #2 / PR #3 is separate experiment work: do not change its files or branch from other tasks.
- Preserve existing LICENSE and research/history. Read [DEVELOPMENT_PLAN](docs/DEVELOPMENT_PLAN.md), [Mentor research](docs/SCIENTIFIC_MENTOR_RESEARCH.md) and [M1/M2 architecture](docs/M1_M2_NUMERICAL_CONTRACT.md). The §25 rank15 correction supersedes the historical rank12 hypothesis for actual PyDOE.
- Production numerical dependency: torch only. PyDOE/NumPy/SciPy are development references, never numerical backends. Prefer native torch primitives; no custom CUDA.
- Reuse COLD/WARM evidence when source/runtime is unchanged; refresh relevant git/gh/Issue/PR state before work. Use EXPLORE → VALIDATE with exact source, real commands and PASS/FAIL/NOT_RUN/BLOCKED.
- Use the designated D: workspace, including Dropbox, for owned scratch/caches/results. Clean only task-owned disposable files and report residue.
- **Current state (2026-10-30): two explicit Scientist-sanctioned Engineer dispatches are active.** (1) M5-A Issue #16 (usability docs/example only; Draft PR #17). (2) M2-B Issue #18 (`c_optimality` criterion; its binding mathematical amendment comment freezes API, oracles C1–C9 and an optional bounded GPU PROBE_ONLY). Nothing else is dispatched. Issue #11 (M3-B1) is completed via PR #12; do not reclaim, rerun or extend it. Issue #13 (M3-B2-P0) and its CPU/GPU probe were explicitly stopped by the Scientist; branch `feat/13-m3b2p0-gather-scores@9cf7d1f` is historical, unreviewed and not permission to continue. No k=2 search, Modified Fedorov, DETMAX or other new implementation/GPU work without a new explicit dispatch Issue.
- Preserve existing historical Issue/PR evidence and branches. The latest Scientist stop overrides older task descriptions and open Issue status. Future authorized Engineer tasks require a fresh scoped instruction and independent review; Engineers never self-merge.
