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
