# W11 held-out H24–H26

Refs #329. Local numerical source `4aad71e39cc1ac2f37c924f763db229dc360a6dd`; second controller source `092127267066f77d898e77ce468110c5fcf236b0`. Python 3.14.6, torch 2.15.0+cpu, CPU float64, seed 0. R seed 20261124.

| Case | TorchOED value | Reference value | Δ |
|---|---|---|---|
| H24 | 6 vertices from frozen call; unfiltered Fraction oracle PASS | mixexp 6 vertices | Hausdorff 7.152557390810443e-08 (coordinates NO MATCH at 1e-12) |
| H25 | D 0.006961468621390328; floor 0.9139763584432278 | AlgDesign D 0.006961468621390309; floor 0.9139763584432251 | reference − TorchOED D = −1.9081958235744878e-17 |
| H26 | D 0.6933612743506347; floor 0.6933612743506348 | skpr NOT_RUN (two captured errors) | N/A |

H24 uses the frozen `mixture_vertices([0.1, 0.1, 0.05], [0.5, 0.7, 0.6])` call without `face_dims`, replacing the earlier row derived from H25 candidates. Every returned coordinate is rationalised with `Fraction(x).limit_denominator(10**6)` and checked within 1e-12; all rows satisfy exact Fraction bounds and sum one, and the unfiltered set and count equal exhaustive bound-pair enumeration. mixexp output carries float32-level precision, explaining the 7e-8 coordinate difference. H25 slogdet/eigenvalue D agreement and rank-six oracle PASS. H26 dense θ inverse-block and Schur logdet agree at 1e-10; `exp(objective) = det M_C` at 1e-9 relative, so `D = exp(objective/6)/16`; objective 14.438307756102468. TorchOED floors are from the returned frozen search results. H25 reference floor evaluated with §80 fixed_ids of length 10 (#334): same-call TorchOED floor 0.9139763584432251 reproduces saved 0.9139763584432278 within 1e-12 (PASS), and reference floor ≥ dense lower check 0.7731778046135993 − 1e-12 (PASS); H26 reference floor stays NOT_RUN: no skpr reference design, both gen_design calls errored (#329).

Per ruling 6986066119, H25 is the second execution of the unchanged deterministic call; the first output was never seen or saved because its subsequent all-fixed certificate call raised. H26 ran once after the prescribed label-preserving cast to int64. No changed seed, model, size, criterion or tuning. A temporary metric self-check compared D directly with exp(objective); corrected to the required normalization and re-scored saved IDs only. A float32 log(16) constant was replaced by float64 before the final D score. No additional search.

Both controller attempts use the pinned #271 command form, CPU R with CUDA hidden: task 2400 s, allocation 2100 s, kernel 120 s, work 1500 s, cleanup 180 s. First source 149a5f40a6f76f19aaa09275eaababdb67127f82: allocation TIMEOUT/ALLOCATION_FAILED; stop PASS, zero assignments, balance 50.87 → 50.87. Second source 092127267066f77d898e77ce468110c5fcf236b0: controller PASS, observed Tesla T4 (R used CPU only), stop PASS, assignments 0 → 0, balance 50.70 → 50.51; usage rate after 0.00/hr. R 4.6.1, mixexp 1.2.7.1, AlgDesign 1.2.1.2, skpr 1.9.2; full sessionInfo() recorded. H26 errors verbatim: "argument must be coercible to non-negative integer" and "undefined columns selected"; stderr warning: "In seq_len(ncol(design)) : first element used of 'length.out' argument". H25 reference metric oracle PASS. Five required metric/oracle checks PASS, zero FAIL; external H24 coordinate mismatch reported as-is. No third attempt. Exact usage and cleanup readback for both attempts are in cleanup.json.

H1–H23 were not rerun; see earlier reports/w*-heldout. No source/tests/docs/website edits, GPU science, timing or speed claims.

measurement only; no tuning followed
