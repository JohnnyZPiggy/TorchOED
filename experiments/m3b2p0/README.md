# Issue13: experimental G-gather scoring (PROBE_ONLY)

Base main d8df1276926af9611f6f52a79653c76ff5d2aa3b includes merged PR12 and
the M3-B2 mathematical gate. This experiment implements **no search**, winner,
swap acceptance, production export or production-file change. Prior API/code,
test/runner/evidence and current SAE metadata stay unchanged as required by the
dispatch. Current experimental authority is Issue13 and its base-update comment.

`gather_two_swap_ratios(F,ids,*,tile_size=16384,max_elements=8_000_000,
form='gather4'|'add_first')` eagerly validates then returns an **iterator of ratio
tensors in canonical order**, not a complete O(pair-count) output. Outgoing
positions i<j, then unselected incoming IDs c<d; IDs/coordinates remain distinct.
CPU/CUDA device follows F.device; float64 only publicly, no implicit transfers.
Initial SPD/pivot checks match the numerical M2 policy, with no ridge. Duplicate,
bad-dtype/device/out-of-range IDs, nonfinite F, initial singular/poorly scaled
Gram, unknown forms, invalid sizes and insufficient budget raise.

With Z=L^-1 F^T, retain G_SS and G_SN. G_NN is retained only when it fits the
budget; otherwise cache bounded incoming row blocks and gather their entries.
Canonical combinatorial ranks use int64 unranking instead of a Python list of
millions of pairs. Pair tiles build the 4x4 G block for `gather4`; `add_first`
uses the always-SPD I2+G_NN pivot and closed-form two-row **solve**, not an explicit
inverse. Delete-first is rejected, never implemented. No Gamma clipping or
positive-ratio-as-SPD proof: this experimental output has **no trial validity
or optimizer certificate**. Nonfinite results are preserved and fail float64
oracle acceptance; they are never substituted with a passing value.

Major tensor element estimate per scorer: fixed=8mp+6p²+4n²+4nq+6m+64; per-pair
tile256. Reserve full q² G_NN only when it leaves room for a pair; otherwise
reserve two cache blocks and shrink tile width. Estimates are not a peak-memory
guarantee for backend/Python/allocator overhead or caller-owned references. The
runner records actual CUDA peak allocation. Iterator/state is task-local, under
no_grad, with no discrete-selection differentiation claim. Public metadata/API
support, GPU performance or cross-version behavior is not asserted.

The CPU runner executes P1–P6. Exact Fraction/Bareiss references stay in their
unchanged tests modules (torch-free). P1 trap/B7/saturated/duplicate/singular
fixtures; P2 rational monomials p4/6/8; P3 all74,448 pairs vs accepted
score_two_swaps VALID records, with exact ordering; P4 independent exact rational
k1 solve/delta; P5 negative/budget/row-cache cases; P6 m200/m1000, n20/p10,
3,060,900 /91,144,900 pairs streamed across tile8192/16384, all indices checked
bitwise and10,000 seeded pairs checked against independently formed direct
Gram determinants. No full91M output is retained or written. P6 p99 error/tol
is over the independent10k reference sample; full-stream equivalence retains
maxima/counts, explicitly not a full independent reference distribution.

Float64 hypothesis (not a rigorous interval):
`abs(computed-reference)<=32*p*eps64*max(1,cond2(M))*max(1,abs(reference))`.
Every exceedance/nonfinite is FAIL; report max and exact sample/full-small p99
observed/tol. No threshold tuning to force PASS. GPU P6float32 is OBSERVED only
via the private shared implementation; it is not enabled in the public entry.
No stochastic restarts are implemented. Threads are process-local probe settings,
recorded along with seeds, actual versions/device and source hashes.

CPU example (from repository root, installed torch/pytest only):

```text
python experiments/m3b2p0/run_probe.py --gates P1 P2 P4 P5 --device cpu --source-sha <exact-commit> --out <owned-D-drive-json>
python -m pytest -q experiments/m3b2p0/test_gather_scores.py
```

All formal execution uses finite outer timeouts recorded in reports/m3b2p0.
Timings are PROBE_ONLY: input H2D, preparation, scoring, reductions and result
D2H included; precomputed reference is outside those times and separately
identified. CPU/other workloads are not standardized performance baselines.

`gpu_driver.py` is a self-contained committed adapter: compressed embedded exact
source bytes with per-file SHA256, including production comparators and MIT
LICENSE; no network code checkout, pip or dependency installation. Its bundle is
verified against committed Git blobs before the live run. It runs P1/P3/P6 on
float64 and P6float32 observations in the **one** pinned-controller Colab session,
reports real GPU identity, synchronized wall times and peak CUDA allocation.
The two required artifact JSON files label top-level PASS as completed
**transport**, with separate scientific_status/probe gate results that can FAIL
or BLOCK. A transport PASS is never scientific acceptance. Each file is bounded
to1MiB and emitted with byte-count/SHA256 receipt, then downloaded/hash-verified.

Pinned SAE controller allocation-seconds bounds the **whole live window**, not
just the new-session call: use900s window,1200s total task,420s max workload,
90s cleanup reserve,30s kernel; actual `new` subprocess is capped120s. Readback
must prove existing positive entitlement and zero assignments before allocation;
stop plus authenticated-zero required afterward, including failures. No new
billing, sessions beyond one, Actions, production edits, Modified Fedorov/DETMAX,
M3-B2 search, speedup/GPU-support claim, self-approval/merge or Issue closure.
