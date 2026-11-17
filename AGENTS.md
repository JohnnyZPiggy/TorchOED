# TorchOED — SAE V3.4 project entrypoint

SAE pin: `4129001fffcb336f5203e4b0d6765d7a0ea3150c` (V3.4 + one-page-per-concept cleanup and Performance Patch workflow, SAE-Framework PRs #74–#76; adopted 2026-11-03; Colab controller and Simplicity Contract unchanged since `5fb26304`). Configuration: [sae.yaml](sae.yaml).
Read the actual [pinned constitution](https://github.com/JohnnyZPiggy/SAE-Framework/blob/4129001fffcb336f5203e4b0d6765d7a0ea3150c/AGENTS.md), assigned role and task-relevant workflows via [.sae/README.md](.sae/README.md); do not silently upgrade or assume links were read.

## Simplicity first (Engineer: read before writing code)

The [Simplicity Contract](https://github.com/JohnnyZPiggy/SAE-Framework/blob/4129001fffcb336f5203e4b0d6765d7a0ea3150c/governance/SIMPLICITY_CONTRACT.md) overrides any older rule or habit that asks for more. The smallest change that meets the Issue's Definition of Done is the correct change.

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
- **Current state:** live ACTIVE/QUEUED dispatch and week progress are in **Issue #83 (Live status)**, edited in place every Architect run; this line is refreshed only at week boundaries. Week 10 (2026-11-17 → 11-24, `docs/WEEK_GOAL.md`): **P2 completion part 2** under VISION.md v4: fractional factorials with fold-over and alias matrix, Plackett–Burman, central composite (incl. blocked) and Box–Behnken, and definitive screening beyond Paley prime orders. **Completeness rule:** every capability is tracked in `docs/CAPABILITY_MATRIX.md`; every feature PR updates its row; a phase ends only when its rows are complete or Scientist-deferred. Plan: W11 mixtures/covariates/compound criteria and blocks + split-plot together, W12 consolidation with ≥ 8 published-example matches, P2+ (full JMP/R parity) weeks 13–15, P2.5 (analysis + TorchOED Studio, VISION v4, Scientist decision 2026-11-17) weeks 17–19, P3 weeks 21–23, P4 weeks 25+; consolidation weeks 12, 16, 20, 24, 28. P1, P1.5 done; P2 core (weeks 6–7), week 8 consolidation (v0.7.0 TestPyPI) and week 9 (blocked and split-plot D/A/I with augmentation, floors and diagnostics; held-out H19–H20 in `reports/w9-heldout`) done. Timing comparisons are per restart (Scientist decision 2026-11-07). Merges follow the ARCHITECT-ACCEPT protocol (Scientist decision 2026-11-09). An Engineer starts only an Issue titled ACTIVE or carrying an Architect START comment. Issue #13 (M3-B2-P0) stays stopped by the Scientist; its branch is historical and not permission to continue.
- Preserve existing historical Issue/PR evidence and branches. The latest Scientist stop overrides older task descriptions and open Issue status. Future authorized Engineer tasks require a fresh scoped instruction and independent review; Engineers never self-merge.
