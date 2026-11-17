# W11 mixture archive retry (#289)

- Sole pinned controller run at `1b7d0a9f6acafd3a0686932e1e8ed3821e1e6f7a`. Observed Tesla T4, CPU R with CUDA hidden. No held-out inputs. Environment otherwise unchanged since #288.
- R 4.6.1, Ubuntu 24.04.4 LTS; exact sessionInfo and packageVersion in sessionInfo.txt. AlgDesign 1.2.1.2 from https://cloud.r-project.org. CRAN mixexp unavailable (verbatim stderr in runtime.json). Archive succeeded: https://cran.r-project.org/src/contrib/Archive/mixexp/mixexp_1.2.7.1.tar.gz. Dependencies lattice 0.23-1 and daewr 1.2-11 from cloud CRAN. r-universe skipped after success. No missing dependency. The inherited installer also passed base grid to install.packages; R skipped it with a warning (verbatim log), no base package installed. This is a procedural deviation from base-package exclusion, not silently repaired after measurement.
- Before allocation the exact Python-decoded archive expression was printed and checked locally: `package_version(sub("[.]tar[.]gz$", "", sub("^mixexp_", "", files)))`. No regex backreference or U+0001. No local R.
- Calls unchanged: `mixexp::Xvert(nfac=3, lc=c(.2,.1,.1), uc=c(.6,.5,.6), ndm=0, plot=FALSE)` and `mixexp::Xvert(nfac=4, lc=rep(.1,4), uc=c(.6,.5,.4,.3), ndm=0, plot=FALSE)`. `set.seed(11); AlgDesign::optFederov(~ -1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3, data=candidates, nTrials=10, criterion="D")`.
- Reproducibility: new optFederov CSV is byte-identical to frozen optfederov.csv. run.py, candidates_r1.csv and optfederov.csv unchanged. Xvert output frozen verbatim; the scorer drops its one dimen≠0 centroid per region, with no rounding or tolerance change.
- Unchanged `run.py`, scored once in local Python 3.14.6 / torch 2.14.1+cpu float64: **R1 NO MATCH** (6/6 vertices; Hausdorff 9.612115315921444e-08), **R3 NO MATCH** (11/11; 4.053554108234124e-08), tolerance 1e-12. **D MATCH**: R 0.004524989087032129, TorchOED 0.004524989087032155, delta 2.6020852139652106e-17, floor 0.9310873256626713. Verdicts are reported as-is; external NO MATCH is not a transport failure.
- Controller transport and stop PASS; authenticated assignments 0→0, balance 51.08→51.08, cleanup_confirmed true. Download SHA-256/size in runtime.json; device/readback in cleanup.json. No remote remains, no retries.
- Eight report files total; only two named tools branches changed (-1 net code line). No shared source changes, no full local pytest required. Final-head CI in PR handoff.

## Installed args() output

```text
[1] ‘1.2.7.1’
function (nfac = 3, uc = c(0, 0), lc = c(0, 0), nlc = 0, lb = c(0, 
    0), ub = c(0, 0), coef, ndm = 0, plot = TRUE, cornerlabs = c("x1", 
    "x2", "x3"), axislabs = c("x1", "x2", "x3"), pseudo = TRUE) 
NULL
[1] ‘1.2.1.2’
function (frml, data = sys.frame(sys.parent()), nTrials, center = FALSE, 
    approximate = FALSE, criterion = "D", evaluateI = FALSE, 
    space = NULL, augment = FALSE, rows = NULL, nullify = 0, 
    maxIteration = 100, nRepeats = 5, DFrac = 1, CFrac = 1, args = FALSE) 
NULL
```
