# Week 6 held-out H11–H13

Issue #199. Measurement source `bfc78a9c62df36e55ecfd5577e908138cc498a08`. One local invocation of `python -B reports/w6-heldout/run.py`, seed 20261112, Windows build 26300, Python 3.14.6, torch 2.15.0+cpu, float64, one thread. Search wall time covers the complete 10-restart call. The [W4](../w4-heldout/) and [W5](../w5-heldout/) records carry H1–H10 forward.

| Case | Frozen result | Required check |
|---|---|---|
| H11 | TorchOED logdet 18.033974267386018, stop and certificate `LOCAL_BLOCK_EXCHANGE`, 0.612391 s wall, 0.061239 s/restart. AlgDesign shared logdet 16.781554674540388, no repeated rows within or across blocks. | PASS. Both shared scores agree with dense projection within 1e-10. TorchOED objective matches its shared score within 1e-9 relative. |
| H12 | 29 runs. Largest absolute second-order column correlation 0.5. | PASS. Conference Gram residual, off-diagonal main-effect Gram and main-by-second-order products all have exact max absolute value 0. |
| H13 | Max VIF 1.233679651608. SPV quantiles (0, .25, .5, .75, 1) = (12.350351855903, 14.123903811334, 15.723343802803, 18.535885142917, 24.836853129133). Objective 33.469839745705, `LOCAL_2_NUMERICAL`, 0.031663 s/restart. | PASS. Dense SPV max error 2.66e-14. Efficiency floor 0.952912723508 meets the 0.603941244972 SPV bound. |

[Results](results.json) include both H11 block assignments as 0-based IDs. The single `optBlock(~quad(A,B,C), withinData=grid, blocksizes=c(6,6,6))` call used the [frozen grid](H11.csv) and R 4.6.1 / AlgDesign 1.2.1.2 on Colab CPU. The pinned controller used a T4 host with CUDA hidden from R. See [sessionInfo](sessionInfo.txt), [runtime](runtime.json) and [cleanup read-back](cleanup.json) (stop PASS, 0 → 0 active assignments). AlgDesign contributes quality measurements only.

measurement only; no tuning followed
