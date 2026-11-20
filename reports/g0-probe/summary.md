# G0 Colab Pro GPU probe — BLOCKED

Issue #385; measured source `d5e594ce9f33d24f73e60edb8fe31d5f1699b237`.
Pinned SAE controller `4129001fffcb336f5203e4b0d6765d7a0ea3150c`; three controller files byte-identical.
Command: `python3 colab_control.py --device H100`, then the single permitted `--device A100` fallback; committed adapter `tools/g0_probe_driver.py`, manifest `tools/crossover_artifacts.json`.
Local preflight PASS. H100: controller status `FAIL`, error `ALLOCATION_FAILED`, new exit 1 (20.359704688 s).
A100: controller status `FAIL`, error `ALLOCATION_FAILED`, new exit 1 (55.199863052 s).
No L4/T4 attempt: one fallback exhausted. No session was created; device name, memory, driver, CUDA and torch version NOT_RUN.
CU before/after: 149.35 / 149.35; rate before/after: 0.00/hr / 0.00/hr; spent 0.00 CU (cap 5).

| k | m | method | seconds | logdet | KW / returned floor |
|---|---|---|---|---|---|
| 3 | 1000 | cpu | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 | 1000 | cuda | NOT_RUN | NOT_RUN | NOT_RUN |

Planned protocol unchanged: float64, seed 0, one restart, one untimed CUDA warm-up and synchronized timer stops.
The existing `--call` implements objective, KW floor and refined floor oracles; all NOT_RUN because allocation failed.
Required tests: 0 PASS, 0 FAIL, 1 NOT_RUN; CPU/CUDA logdet agreement at 1e-9 relative NOT_RUN. No quality or timing claim.
Both cleanup readbacks PASS: stop exit 0, usage exit 0, assignments 0 → 0; fresh usage confirms zero assignments and zero rate.
Next: Architect ruling on allocation/fallback; no further session authorized by this task.

## Retry — 2026-11-19 approximately 21:00 Europe/Zurich — science BLOCKED

Source `3f0c6fd6a8ce7a561f7c18363e19f32a7505c072`; same pinned controller and committed `tools/g0_probe_driver.py` / manifest. Preflight PASS. Original records above are unchanged.

- 2026-11-19T19:56:50Z: H100, controller `FAIL`, error `ALLOCATION_FAILED`, 25.946238784s; cleanup confirmed, assignments 0 → 0; CU 149.35 → 149.35.
- 2026-11-19T19:57:18Z: A100, controller `FAIL`, error `ALLOCATION_FAILED`, 57.508000605s; cleanup confirmed, assignments 0 → 0; CU 149.35 → 149.35.
- 2026-11-19T19:58:16Z: L4, controller `PASS`, error `None`, 48.106131453s; cleanup confirmed, assignments 0 → 0; CU 149.35 → 149.35.

Observed NVIDIA L4, 23034 MiB, driver 580.82.07; torch 2.14.1+cu130, CUDA 13.0, capability 8.9.
Transport PASS; scientific status ERROR. CUDA warm-up, CPU call and CUDA call each exited 1 before measurements: `python -B tools/bench_crossover.py --call 3 1000 {cuda,cpu,cuda}` imports `bench_b0.py`, whose line 18 imports unavailable `pydoe`. Exact error: `ModuleNotFoundError: No module named 'pydoe'`. No dependency installed; no driver/source change or second successful session.

| k | m | method | seconds | logdet | KW / returned floor |
|---|---|---|---|---|---|
| 3 | 1000 | cpu | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 | 1000 | cuda | NOT_RUN | NOT_RUN | NOT_RUN |

Required scientific oracles: 0 PASS, 0 FAIL, 1 NOT_RUN (calls failed before numerical checks); logdet agreement NOT_RUN. Local evidence checks: 6 PASS, 0 FAIL (cleanup, order, artifact error, empty rows, original-record preservation, final usage).
Final usage readback: assignments 0, rate 0.00/hr, balance 149.35 CU. Displayed delta 0.00 CU (provider precision), below 5 CU cap. Downloaded artifact SHA-256 and byte size are in results.json.
Next: Architect ruling on the import/adapter scope before any new science session; retries stopped after successful L4 allocation. No performance or pipeline-proven claim.

## Retry2 — 2026-11-20 02:00 Europe/Zurich — PASS

Source `3602539451eb16fadf7df17114702111e8baca31`; pinned controller and committed G0 adapter unchanged. Local preflight PASS. Previous records above are unchanged.

- 2026-11-20T01:00:21Z: H100, controller `FAIL`, error `ALLOCATION_FAILED`, 17.385575014s; cleanup confirmed, assignments 0 → 0; CU 149.35 → 149.35.
- 2026-11-20T01:00:40Z: A100, controller `PASS`, error `None`, 54.583866271s; cleanup confirmed, assignments 0 → 0; CU 149.35 → 149.35.

Observed device/runtime: {"nvidia_smi": [{"name": "NVIDIA A100-SXM4-40GB", "memory_total_mib": 40960, "driver_version": "580.82.07"}], "torch": {"status": "AVAILABLE", "version": "2.14.1+cu130", "cuda": "13.0", "name": "NVIDIA A100-SXM4-40GB", "capability": [8, 0]}}.

| k | m | method | seconds | logdet | KW / returned floor |
|---|---|---|---|---|---|
| 3 | 1000 | cpu | 0.7006452809999928 | 20.009800061023043 | 0.7894658458649016 / 0.9722818445351031 |
| 3 | 1000 | cuda | 0.98678165299998 | 20.009800061023043 | 0.7894658458649016 / 0.9722818445351031 |

Protocol: float64, seed 0, one restart; one untimed CUDA warm-up, cpu then cuda; synchronized timer stop and transfers included. No PyDOE install. Required numerical checks: 7 PASS, 0 FAIL, 0 NOT_RUN; CPU/CUDA logdet relative 1e-9 agreement: True.
Final usage: assignments 0, rate 0.00/hr, balance 149.35 CU; displayed delta 0.00 CU (provider precision), cap 5. Artifact size/SHA-256 in results.json. Cleanup PASS. No broader performance claim.
