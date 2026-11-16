# W8-K4 published skpr split-plot comparison

**NOT_RUN. W8-K4 remains unmet (0 published examples compared).** The single T4 allocation for source `fe1221c8fe2fdcef921cf4c9d6fe173d26ff41ab` timed out in the Colab CLI `new` call. The pinned controller returned `ALLOCATION_FAILED` after the T4 preflight passed. No session was created, so no package versions, candidate or design CSV, model column count, scores, or delta were observed.

The frozen source is skpr 1.9.2 `gen_design.Rd`, [Examples lines 217–230](https://github.com/cran/skpr/blob/1.9.2/man/gen_design.Rd#L217-L230), copied verbatim into `tools/r_heldout.R` with seed 20261118. The intended 33-run model has columns `1`, Altitude, Range(Close), Range(Far), and Power, with Medium as the effects-coding reference. The example requests 11 whole plots, with skpr assigning their sizes and using its default variance ratio of 1. TorchOED's frozen seed is 0. The comparison and dense GLS assertions require the actual R outputs.

The approved configuration was a T4 host, CUDA hidden from R, GPU unused, no GPU claim. The [Architect ruling](https://github.com/JohnnyZPiggy/TorchOED/issues/217#issuecomment-6920592918) permits exactly one T4 attempt and directs a NOT_RUN verdict if unavailable.

The controller's stop returned PASS and its authenticated read-back confirmed zero active assignments before and after the attempt. See [cleanup.json](cleanup.json) for the controller output.
