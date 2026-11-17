# Week 10 C1/C2 reference freeze — NOT_RUN

Refs #247. Measurement source `5809a8d328ce670b84a1561a64314a2872096075`; SAE controller pin `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. T4 preflight passed. The single allowed allocation failed before a runtime was created: `ALLOCATION_FAILED`, `new` TIMEOUT (operation exit −1; process exit 1). No artifact was downloaded.

Run from the WSL login shell:
~~~sh
python3 /mnt/d/sae-scratch/.sae-247-20261117/controller/colab_control.py --repo-path /mnt/d/sae-scratch/.sae-247-20261117/linux-repo --source-sha 5809a8d328ce670b84a1561a64314a2872096075 --run-id w247-colab-20261117 --out /mnt/d/sae-scratch/.sae-247-20261117/colab --cli colab --device T4 --temp-root /mnt/d/sae-scratch/.sae-247-20261117/temp --driver tools/r_heldout_driver.py --artifacts tools/r_heldout_artifacts.json --task-seconds 2400 --allocation-seconds 2100 --kernel-timeout 120 --work-timeout 1500 --cleanup-seconds 180
~~~

R, FrF2, DoE.base and rsm versions, the installed catalogue citation and first WLP index, and the `7-4.1` cross-check are all NOT_RUN. Catalogue 0/22 rows downloaded, 0 comparisons; PB orthogonality, CCD shapes/alpha, and BBD shapes are NOT_RUN. The local scorer printed these four NOT_RUN outcomes. H21–H23 were not called.

Cleanup PASS: controller stop and fresh authenticated usage each confirmed 0 assignments remaining (0 → 0; credit 51.64 → 51.64). See [controller/runtime](runtime.json) and [cleanup](cleanup.json). No retry per the Architect ruling.
