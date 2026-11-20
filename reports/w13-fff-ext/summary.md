# W13 F3 R support reference (#409)

One R oracle run [44656949570](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44656949570), source `e4fad1e38ed6ff6a707abd95ea90ca2dccdffce4`, base `879a7f23152b5711872bebf651b4d1765cd688ef`.
R 4.5.2; support 0.1.7. Exact installed dependency versions in `runtime.json`; workflow session in `sessionInfo.txt` (fresh R process). No dependency pin beyond support was needed. Raw full-precision design and installed Rd2ex text are in `raw.json`.
Freeze once: CPU float64, Sobol 1024 x 2, scramble=True, seed=0; x1+x2<=1 gives 511 rows. All ten default FFF design rows belong to that frozen sample.
R call: seed=20261119, sp(n=10,p=2,dist.samp=F,par.flg=FALSE), defaults otherwise (including scale.flg=TRUE: internal unit-variance scaling, returned points back on original scale). No extra target argument was needed. No tuning or second dispatch.

| Shared metric | TorchOED | R support | Better |
|---|---|---|---|
| minimum distance | 0.18386538194555926 | 0.18615522639545923 | R (higher) |
| phi15 | 5.523994155929388 | 5.645961243284867 | TorchOED (lower) |
| MaxPro psi | 26.20592530774234 | 133.56149083468995 | TorchOED (lower) |
| CD2 | 0.22953686789709923 | 0.2774727717629948 | TorchOED (lower) |
| energy E | 0.4310371650458609 | 0.4232797029915171 | R (lower) |

R max(x1+x2-1) = -0.05868936674633085 (feasible). Support points need not be sample rows. All metrics use shared TorchOED diagnostics / torch.cdist on original coordinates. Energy drops the same sample-only constant for both designs; no combined verdict.
Published count: 14 -> 14. First sp.Rd example (source line 66; installed Rd2ex line 15), n=25,p=2, independent normal target, is not comparable to FFF on a subset of [0,1]^2.
**NOT_RUN example:** Rd2ex preserves dontrun examples as ##D comments. The executed run.R eval parsed those comments, so no example design was produced (raw example design is empty). The main frozen-sample sp call did complete. A second dispatch is prohibited by #409; BLOCKED posted for Architect ruling on the missing example execution. Raw output and run.R are preserved as executed.
Reproduce: `PYTHONPATH=src python reports/w13-fff-ext/score.py`; byte-identical results verified on final HEAD. Ten report files; only F3 Ext matrix cell, one index row and one dev-only package pin changed. No src/tests/website/pyproject edits, Colab or held-out work.
