# Week 10 C1/C2 reference freeze

Refs #257. This extends the five rsm cases frozen in [#254](https://github.com/JohnnyZPiggy/TorchOED/pull/254). Measurement source `91251ddd8741766d844a30558dcdb58c0b143639` and controller pin `4129001fffcb336f5203e4b0d6765d7a0ea3150c`.

~~~sh
python3 "/mnt/d/sae-scratch/.sae-257-20261117/controller/colab_control.py" --repo-path "/mnt/d/sae-scratch/.sae-257-20261117/linux-repo" --source-sha 91251ddd8741766d844a30558dcdb58c0b143639 --run-id "w257-colab-20261117" --out "/mnt/d/sae-scratch/.sae-257-20261117/colab" --cli colab --device T4 --temp-root "/mnt/d/sae-scratch/.sae-257-20261117/temp" --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 120 --work-timeout 1500 --cleanup-seconds 180
~~~

- The sole run began 2026-11-17 13:48:39 UTC and exited 0. A Tesla T4 hosted CPU R with CUDA hidden. Stop passed and a fresh authenticated read-back confirmed 0 → 0 assignments and 51.41 → 51.32 compute units. See [runtime](runtime.json) and [cleanup](cleanup.json).
- [Installed versions and help](sessionInfo.txt) record R 4.6.1, FrF2 2.3.5, DoE.base 1.2.5 and rsm 2.10.6. The FrF2 help states the WLP starts at word length **1**. Installed catalogue citation: Chen, J., Sun, D.X. and Wu, C.F.J. (1993) A catalogue of 2-level and 3-level orthogonal arrays. _Int. Statistical Review_ *61*, 131-145.
- [Catalogue](frf2_catlg.csv): **22/22** rows frozen. The shared exact-integer WLP scorer reported **7/22** full-vector matches. Six of the 15 mismatches have surplus zero entries past length k in the catalogue. Nine have nonzero TorchOED counts beyond the stored seven entries. The scorer prints both vectors for each mismatch. FrF2 stores `7-4.1` as `[0, 0, 7, 7, 0, 0, 1]`.
- [FrF2 PB](frf2_pb.csv): exact Gram `XᵀX = N·I` for N = 12, 20 and 24. [rsm BBD](rsm_bbd67.csv): observed k = 6 at 51×6 and k = 7 at 59×7.
- The downloaded [CCD](rsm_ccd.csv) and [BBD](rsm_bbd.csv) CSVs both matched the previous committed bytes. No K6 reference branch returned NOT_RUN. H21–H23 were not called.
- Local CPU re-score (Refs #261): `python reports/w10-rref/run.py` exited 0. FrF2 PB matched `plackett_burman` row-for-row at N = 12, 20 and 24 (3/3), bringing C2 to 8 matching cases. The catalogue matched 22/22 under §34.3 and 7/22 under the original full-vector check.
