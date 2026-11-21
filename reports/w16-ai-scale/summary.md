# W16 A/I CPU scale — #527

Numerical source: `b4e5c2ddff308aec916d72750ed9901ca922bfa2` → `9f700e7ccc0e8163d119bd3f67adaea4b5f5828a`. Windows 11 / AMD Ryzen 7 4800H (8 cores, 16 logical CPUs), CPU float64, Python 3.14.6, torch 2.14.1+cpu, one Torch thread. k=5 full quadratic p=21, n=42, candidate and design seed 0, one restart, max_iter=100. Explicit `max_elements=20_000_000` throughout: default 1M rejects the larger cells; no default-capacity improvement claim.

Diagnosis was posted before numeric changes in [#527](https://github.com/JohnnyZPiggy/TorchOED/issues/527#issuecomment-7060464112). Pair scoring takes 92–97% of the four completed baseline cells. Quadratic workspace reserves disable screening, reverting to C(42,2) C(m−42,2) pairs (394,684,983 at m=1000; 42,684,852,483 at m=10000). Bounded gather4 gathers p-length vectors per pair plus a fixed 4×4 solve; see M3B2_K2_SEARCH_GATE §3 and W1_GENERIC_CRITERIA_GATE §§1–2.

Fix: reuse W1_GENERIC_CRITERIA_GATE §2e removal-tangent enumeration with only selected G/Q rows and workspace-bounded outgoing blocks, O(n²q log q) plus survivors; existing §2d masking, closed screen, gather4, verification, tie/resume order and certificates remain. No neighborhood is omitted.

| m | A before s | A after s / floor | I before s | I after s / floor |
|---|---:|---:|---:|---:|
| 1000 | 4.895 | 5.556 / 0.932253010 | 15.783 | 20.497 / 0.942217585 |
| 2000 | 17.253 | 22.514 / 0.943333006 | 22.440 | 30.004 / 0.933565411 |
| 5000 | 600.014 TIMEOUT | 47.500 / 0.932813117 | 600.018 TIMEOUT | 162.277 / 0.939162225 |
| 10000 | 600.007 TIMEOUT | 600.003 TIMEOUT / N/A | 600.012 TIMEOUT | 333.367 / 0.933883894 |

Quality: 4/4 completed-baseline cells have identical IDs, objectives and floors; six m=200 pre-change objectives (A/I, seeds 0–2) pass at 1e-12 relative. New low-workspace tests compare full-scan winners and independent dense information solves. Timed-out calls have no returned objective/floor.

Target m=10000 ≤120 s: NOT MET. Remaining bottleneck is pair enumeration/screening/rescoring; no global optimum or certificate for timed-out A is claimed; completed I retains LOCAL_2_NUMERICAL and its floor. m=5000 now finishes, but old calls were censored, so large-cell objective comparisons are unavailable.

Empirical baseline m=1000→2000 exponents log2(t2/t1): A 1.82, I 0.51; m=2000→5000 censored lower bounds A >3.87, I >3.58. These are observations, not asymptotic fits. Phase times for completed calls are in results.json; interrupted cProfile cumulative totals are non-additive and retained separately as invalid, with phase times null. Certificate was not reached for timeouts.

B0 attempt 3: D results identical on all 8×40=320 recorded TorchOED outputs; recorded aggregate timing cells ≤1.10×: 8/8, ratios 0.9447–1.0318. Full unmodified tools/bench_b0.py; intended base/head/base/head sequence, with the recovery deviation below; runner and per-case numbers in results.json. B0 uses the existing PyDOE dev environment (torch 2.15.0+cpu), separately from profiles. First two timing attempts failed under concurrent load and remain recorded; none selected by fastest result.

Reproduce profiles: `python tools/profile_ai.py M A_or_I OUTPUT 600`; run at each recorded source SHA. Baseline and after profiles overlapped with each other and, for early cells, local checks; wall times are measured under that disclosed load. B0 attempt 3 ran after all profiles/checks ended. No GPU, R, held-out, new API, dependency, matrix or CHANGELOG work.

Recovery limitation: stale memory led to an accidental duplicate baseline at 09:46:15 UTC, overlapping the last 20 s of head2 and replacing base1. The duplicate driver was stopped before any further head run; its baseline child completed. The final aggregate uses that replacement baseline and original head1/base2/head2. It is not a fully isolated acceptance run; all observed timing failures remain recorded and no further retry was made.

## Isolated re-measurement (#532)

One attempt, sequential base/head/base/head; no concurrent local benchmark, test or build. Base `cd142da7a0d2ebb71ee04e7003a8a1ec471fcf7d`; head `23c7ca126368cdab6a774c72b6ff6546d2c10cde`. Same frozen setup, unmodified `tools/profile_ai.py`; per-call new-branch boolean from a monkeypatch counter in the driver. Each value below is the median of two calls; ratios use unrounded medians.

Machine: Windows-11-10.0.26300-SP0, AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Python 3.14.6, Torch 2.15.0+cpu, float64, one Torch thread; k=5, p=21, n=42, seed0, one restart, max_iter=100, max_elements=20,000,000.

| m | Criterion | Base total s | Head total s | Total ratio | Base pair s | Head pair s | Pair ratio | #531 branch fired |
|---|---|---|---|---|---|---|---|---|
| 1000 | A | 4.665 | 4.680 | 1.0031 | 4.332 | 4.323 | 0.9979 | False |
| 1000 | I | 15.699 | 15.517 | 0.9884 | 15.307 | 15.127 | 0.9882 | False |
| 2000 | A | 17.388 | 17.462 | 1.0042 | 16.715 | 16.769 | 1.0032 | False |
| 2000 | I | 22.879 | 22.653 | 0.9901 | 22.266 | 22.080 | 0.9917 | False |

Pair-phase cells >1.10: 0/4; no source or test change and Step 2 does not apply. All 16 measured calls reproduce #531 ids and objectives exactly (16 PASS, 0 FAIL). Base/head efficiency floors agree exactly in all 16 calls, but the frozen #531 I m=1,000 floor is 0.9422175849514731 while all four new calls give 0.9422175849514733 (delta 2.22e-16). Frozen-floor equality is 12 PASS, 4 FAIL: strict DoD item 3 is BLOCKED pending Architect disposition; no tolerance is silently substituted. The overlapping-load slowdown in #531 was not reproduced. No new m=5,000 or m=10,000 calls, nor B0 rerun (conditional Step 2 only). Earlier report contents are preserved.
