# ENGINEER_HANDOFF — Issue #18 / M2-B

**CPU_PASS / GPU_SCOPE_BLOCKED / DRAFT_FOR_INDEPENDENT_REVIEW / NO_SELF_MERGE**

Base: `2bab7cf6190e6a57a12f6502a84a7ffbf34e9952`.
Branch: `feat/18-m2b-c-optimality`.
Frozen SAE: `d84942e4890d9837ae0e6cbb379b85c3a95aba84`.
CLAIM: https://github.com/JohnnyZPiggy/TorchOED/issues/18#issuecomment-6529273576
Final full CPU tested commit: `3c3a50df8e800e0537456e3434958ca2af7f2ff2`.
Latest exact CPU EXPLORE / GPU adapter: `04cfaa816a8d8b7065f8c1d813f811228d39bba7`.
Final evidence-only HEAD is recorded in the PR comment to avoid self-reference.
Runtime/tests/oracle/CPU runner at these two commits are byte-identical; only GPU
execution tooling changed. Wheel was built at `8819053de506902ca37984d4917a6e649b3fd35d`;
all package runtime files/pyproject/README are identical to final tested source.
SHA256SUMS covers every changed file except itself, using canonical Git bytes.
The Draft PR Files changed view is the complete diff.

## Recovery and implementation

Issue body and both binding amendments fully read; latest Scientist scope
authorizes only c_optimality and G1-G5, not stopped Issue13 or queued Issue20.
Current main/base, existing PRs/branches and CLAIM conflicts checked before CLAIM.
Project AGENTS/sae config and numerical/API contracts read; immutable SAE
constitution/role/COLD/WARM/task/lean/security evidence reused, Engineer and
compute discovery/controller/GPU policy reread. Existing installed environments
and audited CLI0.7.4 reused. No silent framework upgrade.

Only new public API is `c_optimality(M,c,*,max_elements=1_000_000)` in criteria.py
and root __all__. Same M2 numerical symmetry/tau/pivot guard; matched dtype/device
and shapes, finite/nonzero contrast, budgets before factorization; triangular
solve and negative squared norm, existing finite-output guard. No inverse,
ridge, clipping, generalized inverse or broadcasting. The singular-estimable
contrast limitation is explicit in docs/M2B_API.md. No search API or new backend.
Coverage ledger updates only the existing c_optimality row and scoped counts
(11 VERIFIED,7 PLANNED,73 DEFERRED), not general PyDOE equivalence.

## CPU correctness and packaging

PASS: C1-C9 independent stdlib Fraction Gauss-Jordan references, diagonal/line
fixtures,24 seeded rational SPD cases for unit/A identity and rank-one updates,
rational rotation, homogeneity, analytic gradients and float64 gradcheck, matched
batching, dtype/shape/device/nonfinite/zero/singular/ill-conditioned/overflow and
budget-before-factorization negatives. Production outputs never define expected
oracle values. float64 atol/rtol1e-9, float32 1e-5; gradcheck eps1e-6,atol1e-5,
rtol1e-3. Final errors in validate-repair/oracle-errors.json: float64 max2.8422e-14,
float32 max1.90735e-6 (separate dtype records). C9 acceptance/rejection is in JUnit.

Full pytest PASS:78 tests,0 failed/skipped,5.75s at full tested source.
Latest adapter source EXPLORE PASS:13 tests,3.51s; numerical files unchanged.
Windows Python3.14.6,torch2.14.1+cpu. Two fresh D-drive venvs inherit existing
dependencies via --system-site-packages: nonhermetic, no cross-platform claim.
Editable install, wheel build, separate wheel install and outside-checkout
13-export/version/c-criterion import smoke PASS. No dependency/version changes.
Wheel SHA256: `c2019d8fb868f152456e1f0ea425cd4222441170d2c1c525ca2884afdbd5acd5`.

Actual runner: `<VENV>/Scripts/python.exe -B tools/validate_m2b.py --output <TASK_EVIDENCE>/validate-repair`.
EXPLORE adds --explore. Each pytest subprocess240s; install/build/venv operations
120s, outside import60s. Full exact argv/exits/durations in install/install-commands.json,
validation.json and logs/JUnit, with private roots replaced by placeholders.
Build: `python -m pip wheel --no-deps --no-build-isolation --wheel-dir <WHEEL> .`;
editable: `python -m pip install --no-deps --no-build-isolation -e .`.

## GPU evidence and unresolved scope

**Observed T4 aggregate PASS:**63 G2 configurations,63 G4 configurations and126
G5 timing records across p=3,10,30,100. Every G2 has both G5 dtypes measured.
Final peak fractions:0.699675,0.699836,0.699051,0.699852 respectively. Main matrix
and continuations preserve their own source IDs and old failure statuses;
aggregate success does not mean earlier interrupted runs individually passed.

All live sessions were sequential via unchanged pinned controller, with fresh
authenticated-zero/positive-existing-entitlement checks before each allocation.
Controller and adapter/source hashes, bounds and exact sanitized argv are under
gpu/ and compute-provenance.json. No purchase, top-up, login, provider patch or
Actions. Controller supports only T4 requests: **L4/G4/H100/A100 BLOCKED**, exact
reason PINNED_CONTROLLER_ARGPARSE_CHOICES_T4_ONLY. CLI help lists those types;
account availability is NOT_TESTED, not claimed unavailable. A compatible pinned
controller interface is needed before the full all-device amendment can pass.

Real observed device: Tesla T4 (15,637,086,208 bytes), torch2.11.0+cu130,
CUDA13.0,driver580.82.07,Python3.13.16. Every downloaded artifact is retained
byte-identical, source-checked and byte-count/SHA256 verified by controller.
Artifact status PASS means delivery only; scientific_status is separate.

Latest run (no missing result promoted to PASS):

```json
{
  "transport_status": "PASS",
  "scientific_status": "PASS",
  "G1": "PASS",
  "G2_configs": 1,
  "G4_configs": 1,
  "G5_measurements": 2,
  "p_observed": [
    100
  ],
  "G5_p_not_run": [
    3,
    10,
    30
  ],
  "active_stage": "COMPLETE",
  "failure_type": null,
  "failure": null,
  "peak_fraction_by_p": {
    "3": 0,
    "10": 0,
    "30": 0,
    "100": 0.699851747725301
  },
  "G2_max_relative_error": 7.208224373969749e-16,
  "G4_max_relative_inf_error": 2.465642041000594e-15
}
```

Observed Tesla T4 break-even from the full-matrix VM (source3bcdf16):

| p / dtype | First measured B, no transfer | First measured B, with transfer |
|---|---|---|
| 3/float64 | 1024 | 1024 |
| 3/float32 | 1024 | 1024 |
| 10/float64 | 256 | 256 |
| 10/float32 | 256 | 256 |
| 30/float64 | 32 | 32 |
| 30/float32 | 32 | 32 |
| 100/float64 | 4 | 4 |
| 100/float32 | 4 | 4 |


This is the first winning measured B in that run, not a stable crossover.
Tail runs have their own CPU baselines and do not redefine this prefix result.
The combined per-configuration coverage and four memory targets are in
t4-aggregate.json; summary.json records its status without promoting blocked
other GPU models. Main/tail/final adapters have identical numerical/test bytes
and seed formula, with each row retaining its source/run.

See summary.json for all runs, G3 rejection rates and every disagreement seed,
float32 sampled accuracy breakdown vs guard rejection, and break-even B per
measured p/dtype. Raw artifacts contain every measured configuration. G1 includes
CUDA float64 C1-C9 and gradcheck. G4 analytic comparison uses independent CPU LU
reference in the latest driver while the actual forward/backward remains CUDA.
G2 CPU sample uses all rows when B<10k and >=10k score rows once available;
float32 normalized max/p99 are OBSERVED, not float32 acceptance claims.
The large-batch G4 metric is relative infinity norm (not per-entry division at
near-zero gradients), threshold1e-8. No tolerances were loosened to obtain PASS.

G5 uses the same Colab VM's CPU (4 torch threads, model NOT_RECORDED), warmup3,
5 synchronized GPU repeats, medians with/without H2D/D2H; exact peak allocation
per recorded configuration. G2 memory target includes diagnostic/autograd
workspace; last batch interpolates toward70% when doubling would overshoot.
G5 peak reset is per dtype but includes resident fp64/fp32 fixtures. First observed
break-even is not a guaranteed stable crossover or a universal speedup claim.
Unmeasured configurations are NOT_RUN; incomplete large gates are BLOCKED.

## Failures and repairs (all retained)

1. Initial CPU full test:77 passed,1 failed because base environment had no
   torchoed distribution metadata. Installed editable in task venv, then78 PASS;
   no production repair. Original failure JUnit/log under validate/.
2. Initial preflight: PREFLIGHT_COMMAND_FAILED/source_verification. WSL Git could
   not resolve Windows worktree .git target. Created task-owned detached source
   clone on D outside Dropbox at the same SHA; preflight passed thereafter.
3. First T4: G1 PASS10; G3 completed;15 G2/G4 configurations and30 G5 records.
   CUDA eigvalsh workspace OOM at p3/B32768 after measured57% peak atB16384.
   Partial artifact preserved; final batch sizing now uses measured memory and
   interpolates instead of blindly doubling beyond capacity.
4. Canonical bundle audit found LICENSE CRLF bytes differed from its Git blob.
   Changed assembler to exact index blobs; all14 entries now match Git exactly.
   Runtime math had matched; no license or production file was altered.
5. Second T4: controller work exec425s external TIMEOUT, no artifact downloaded.
   Scientific outcome unobserved; cleanup PASS. Isolated matrix child now has350s
   bound and active-stage checkpoints to retain partial results within420s transport.
6. Third T4: G1 PASS10;38 G2,37 G4,74 G5 records. At G4/p30/B32,
   cudaErrorNoKernelImageForDevice reported at CUDA torch.linalg.solve reference
   (asynchronous attribution caveat retained). Latest driver uses CPU LU reference
   and explicit GPU synchronization; algorithm and thresholds unchanged.
7. Fourth T4 completed62 G2/G4 configurations and122 G5 records across all four p;
   finite340s matrix window ended at G5/p100/B16384/float64 (elapsed343.04s).
   Partitioned remaining p100/B>=16384 into a separate committed tail adapter;
   same seeds and numerical source; earlier successful configurations are reused.
8. Fifth T4 tail completed p100/B16384 and B18150 G2/G4 plus three G5 records;
   finite340s window ended at G5/p100/B18150/float32. Final continuation starts
   only B18150 and uses400s matrix/410s child within420s pinned workload bound.
   This is the third timeout-class execution repair (isolation, partition, final
   narrowed configuration); no further automatic timeout repair is attempted.
9. Ad hoc coverage audit commands initially failed from Windows encoding/regex
   escaping and exact-match counting of annotated status cells; final structured
   cell parsing PASS91 rows. No scientific test was bypassed or changed for this.

Latest run's failure, if any, is recorded above and verbatim in its artifact or
controller report. No retry exceeds3 meaningful repairs for a failure class;
no quota bypass, unbounded retry or background continuation is installed.

## Boundaries, storage and next action

README/release-readiness/pyproject/SAE pin/root license, original tests, historical
reports and Issue13 branch are unchanged. Old regression cases run only as part
of bounded full pytest; no old milestone is reclaimed or standalone probe rerun.
Issue11 COMPLETE; Issue13 STOPPED; Issue20 QUEUED. No M3 search, Modified Fedorov,
DETMAX, other criteria, release/tag/publication, self-approval or self-merge.

Real workspace is the Scientist-designated D-drive Dropbox root (private prefix
redacted in public logs). Build temp: D:/sae-m2b-20261030-build; transport temp:
D:/sae-m2b-20261030-temp. No large C:/WSL virtual-disk dataset/cache created.
Automatic approval review rejected deletion of the owned transport source clone
and pip build cache, returning only `blocked by policy`; that command did not run.
No deletion bypass was attempted. Local scratch/cache is retained, not reported
as zero. Actual venv/wheel/log/source/temp counts are in residue.json.
Raw local originals remain; raw-public-provenance.json records hashes
and transformations. All allocated sessions have explicit successful stop plus
authenticated-zero receipts; preflight-only records did not allocate sessions.

NOT_RUN: unsupported GPU models, uncompleted GPU configurations, minimum-version
and cross-platform matrices, formal release, Actions and generalized singular
c-optimality. No universal GPU support/throughput claim.

Next bounded action: Architect independently review exact final HEAD, C1-C9,
source identity, GPU partial/blocker evidence and manifest; resolve the pinned
controller's multi-device interface before claiming complete G1-G5 across all
available models. Engineer leaves this one Draft PR unmerged. Refs #18.
