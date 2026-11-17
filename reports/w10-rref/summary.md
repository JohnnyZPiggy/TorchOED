# Week 10 C1/C2 reference freeze — partial

Refs #247. Measurement source `5809a8d328ce670b84a1561a64314a2872096075`; controller pin `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. T4 preflight passed; both attempts used the WSL login shell.

~~~sh
python3 /mnt/d/sae-scratch/.sae-247-20261117/controller/colab_control.py --repo-path /mnt/d/sae-scratch/.sae-247-20261117/linux-repo --source-sha 5809a8d328ce670b84a1561a64314a2872096075 --run-id w247-colab-20261117 --out /mnt/d/sae-scratch/.sae-247-20261117/colab --cli colab --device T4 --temp-root /mnt/d/sae-scratch/.sae-247-20261117/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 120 --work-timeout 1500 --cleanup-seconds 180
python3 /mnt/d/sae-scratch/.sae-247-20261117/controller/colab_control.py --repo-path /mnt/d/sae-scratch/.sae-247-20261117/linux-repo --source-sha 5809a8d328ce670b84a1561a64314a2872096075 --run-id w247-colab-retry1-20261117 --out /mnt/d/sae-scratch/.sae-247-20261117/colab-retry1 --cli colab --device T4 --temp-root /mnt/d/sae-scratch/.sae-247-20261117/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 120 --work-timeout 1500 --cleanup-seconds 180
~~~

- Initial attempt, 11:51:26 UTC: exit 1, `ALLOCATION_FAILED`, allocation `new` TIMEOUT (operation exit −1), no runtime or download. Stop passed; authenticated assignments 0 → 0, 51.64 → 51.64 units.
- Authorized retry, 12:49:20 UTC: controller exit 0 on a Tesla T4 host with CUDA hidden from R. Stop passed; authenticated assignments 0 → 0, 51.48 → 51.41 units. See [runtime](runtime.json) and [cleanup](cleanup.json).
- [R 4.6.1 and rsm 2.10.6](sessionInfo.txt). The CPU float64 scorer verified all five shapes. [CCD](rsm_ccd.csv) A 14×2, B 30×4, C 46×5, with α = 1.4142135623731, 2 and 1, respectively. [BBD](rsm_bbd.csv) k = 3: 15×3; k = 5: 43×5.
- FrF2 and DoE.base versions, installed citation and WLP start index, `7-4.1`, and FrF2 PB are NOT_RUN. Catalogue 0/22 rows and zero WLP comparisons. The FrF2 install exited 1. Its diagnostic began `configure: error: Header file gmp.h not found; maybe use --with-gmp-include=INCLUDE_PATH`. The exact dependency errors and K6 output are in [runtime](runtime.json). H21–H23 were not called.
