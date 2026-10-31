# TorchOED — SAE V3.4 project entrypoint

SAE pin: `5fb26304e4c85183aa215e6ad4db9ced928cab12` (V3.4 + simplified Colab controller with G4 → RTX PRO 6000 family match, SAE-Framework PRs #71 and #73; adopted 2026-10-30). Configuration: [sae.yaml](sae.yaml).
Read the actual [pinned constitution](https://github.com/JohnnyZPiggy/SAE-Framework/blob/5fb26304e4c85183aa215e6ad4db9ced928cab12/AGENTS.md), assigned role and task-relevant workflows via [.sae/README.md](.sae/README.md); do not silently upgrade or assume links were read.

## Simplicity first (Engineer: read before writing code)

The [Simplicity Contract](https://github.com/JohnnyZPiggy/SAE-Framework/blob/5fb26304e4c85183aa215e6ad4db9ced928cab12/governance/SIMPLICITY_CONTRACT.md) overrides any older rule or habit that asks for more. The smallest change that meets the Issue's Definition of Done is the correct change.

1. **Validate only at trust boundaries** (CLI args, files, provider output), only the fields you use. Trust internal code: no re-checking shapes/dtypes/types your own code produced. Let torch and Python raise.
2. **Catch an exception only if you then do something different.** No catch-and-rename, no custom error-code taxonomies.
3. **No speculative code:** no "future" abstractions, flags, fallbacks, length caps, hand-written JSON/schema validators, `ast`/regex enforcement of conventions.
4. **Tests:** one per changed behavior, plus an independent oracle (e.g. dense `torch.linalg`, PyDOE reference) for numerical claims. No randomized/adversarial suites unless the Issue asks. Test lines ≲ source lines.
5. **Evidence:** Git SHA is the integrity layer — never hash tracked files. Handoff = one PR comment; ≤ 3 evidence files for code, ≤ 10 for a GPU run.
6. **Budget:** every Issue states a size budget (S ≤ 50 / M ≤ 200 / L ≤ 500 net lines) and a *Not required* list. Over 1.5× → stop, ship the Definition of Done, list extras as suggestions.

```python
# Not this
def d_criterion(X):
    if not isinstance(X, torch.Tensor): raise TypeError("X must be a tensor")
    if X.ndim != 2: raise ValueError(f"bad shape {X.shape}")
    try:
        return torch.linalg.slogdet(X.T @ X).logabsdet
    except RuntimeError as e:
        raise CriterionError("D_FAILED") from e

# This
def d_criterion(X):
    return torch.linalg.slogdet(X.T @ X).logabsdet
```

## Scope and roles

- Scientist sets the objective and can redirect/stop work; Architect/Mentor owns specifications, mathematics and independent review.
- Engineer implements/tests on independent branches and submits evidence; **never self-approves or merges**, never writes protected main or force-pushes.
- M0 Issue #1 historically permitted only the installable torch-only spine, frozen upstream inventory, minimal SAE adoption and CPU smoke. Planning documents alone do not dispatch algorithms. M1 Issue #5 is merged. M2-A Issue #7 is merged. M3-A Issue #9 is merged. Historical Issue #11 authorized CPUfloat64 exact-ID no-repeat simultaneous k=2 scores and oracles, with no search loop; PR #12 has since merged. M3-B2-P0 Issue #13 was stopped by the Scientist. Modified Fedorov, DETMAX, and k-swap search remain unassigned.
- E0 Issue #2 / PR #3 is separate experiment work: do not change its files or branch from other tasks.
- Preserve existing LICENSE and research/history. Read [DEVELOPMENT_PLAN](docs/DEVELOPMENT_PLAN.md), [Mentor research](docs/SCIENTIFIC_MENTOR_RESEARCH.md) and [M1/M2 architecture](docs/M1_M2_NUMERICAL_CONTRACT.md). The §25 rank15 correction supersedes the historical rank12 hypothesis for actual PyDOE.
- Production numerical dependency: torch only. PyDOE/NumPy/SciPy are development references, never numerical backends. Prefer native torch primitives; no custom CUDA.
- Reuse COLD/WARM evidence when source/runtime is unchanged; refresh relevant git/gh/Issue/PR state before work. Use EXPLORE → VALIDATE with exact source, real commands and PASS/FAIL/NOT_RUN/BLOCKED.
- Use the designated D: workspace, including Dropbox, for owned scratch/caches/results. Clean only task-owned disposable files and report residue.
- **Current state (2026-10-31): one Engineer dispatch is active: M5-B Issue #31** (CPU CI on Linux+Windows, Python 3.11/3.13, plus clean wheel install; GitHub-hosted runners, supersedes older "no Actions" lines). M3-B2 Issue #28 `k2_exchange` is merged via PR #30 (S2: 205/205 2-local, 145 global). M5-A2 Issue #20 is merged via PR #25. M2-B Issue #18 (`c_optimality`) is complete: CPU via PR #21, multi-GPU rerun via PR #26, evidence in `reports/m2b-multigpu/`. M5-A Issue #16 is merged via PR #17. Issue #11 (M3-B1) is completed via PR #12; do not reclaim, rerun or extend it. Issue #13 (M3-B2-P0) and its CPU/GPU probe were explicitly stopped by the Scientist; branch `feat/13-m3b2p0-gather-scores@9cf7d1f` is historical, unreviewed and not permission to continue. No Modified Fedorov, DETMAX, k≥3 or other new implementation/GPU work without a new explicit dispatch Issue or START comment.
- Preserve existing historical Issue/PR evidence and branches. The latest Scientist stop overrides older task descriptions and open Issue status. Future authorized Engineer tasks require a fresh scoped instruction and independent review; Engineers never self-merge.
