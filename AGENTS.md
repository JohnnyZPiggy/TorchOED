# TorchOED — SAE V3.3 project entrypoint

SAE pin: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`. Configuration: [sae.yaml](sae.yaml).
Read the actual [pinned constitution](https://github.com/JohnnyZPiggy/SAE-Framework/blob/d84942e4890d9837ae0e6cbb379b85c3a95aba84/AGENTS.md), assigned role and task-relevant workflows via [.sae/README.md](.sae/README.md); do not silently upgrade or assume links were read.

## Scope and roles

- Scientist sets the objective and can redirect/stop work; Architect/Mentor owns specifications, mathematics and independent review.
- Engineer implements/tests on independent branches and submits evidence; **never self-approves or merges**, never writes protected main or force-pushes.
- M0 Issue #1 historically permitted only the installable torch-only spine, frozen upstream inventory, minimal SAE adoption and CPU smoke. Planning documents alone do not dispatch algorithms. M1 Issue #5 is merged. Explicit Issue #7/user dispatch permits only M2-A exact_information, weighted_information, d_optimality, a_optimality and e_optimality with A1–A12 CPU oracles; other M2 and all M3 remain out of scope.
- E0 Issue #2 / PR #3 is separate experiment work: do not change its files or branch while executing M2-A.
- Preserve existing LICENSE and research/history. Read [DEVELOPMENT_PLAN](docs/DEVELOPMENT_PLAN.md), [Mentor research](docs/SCIENTIFIC_MENTOR_RESEARCH.md) and [M1/M2 architecture](docs/M1_M2_NUMERICAL_CONTRACT.md). The §25 rank15 correction supersedes the historical rank12 hypothesis for actual PyDOE.
- Production numerical dependency: torch only. PyDOE/NumPy/SciPy are development references, never numerical backends. Prefer native torch primitives; no custom CUDA.
- Reuse COLD/WARM evidence when source/runtime is unchanged; refresh relevant git/gh/Issue/PR state before work. Use EXPLORE → VALIDATE with exact source, real commands and PASS/FAIL/NOT_RUN/BLOCKED.
- Use the designated D: workspace, including Dropbox, for owned scratch/caches/results. Clean only task-owned disposable files and report residue.
- Current M2-A dispatch is **CPU only**: no GPU/Colab, Actions, paid services or global configuration changes. Other work requires its own scoped dispatch; do not infer compute authority from a plan.
- Submit one Draft PR with Refs #7 and ENGINEER_HANDOFF. No self-merge; keep evidence sanitized and provide a concrete next handoff.
