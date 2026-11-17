# W10 C3 — daewr definitive screening (Refs #266)

The single CPU R reference attempt used TorchOED source `13a444e3ccd77d8b18efc557df854eb8757c9cb0` and SAE controller pin `4129001fffcb336f5203e4b0d6765d7a0ea3150c`. The T4 allocation command `colab new -s sae-w266-colab-20261117 --gpu T4` timed out at its 120 s cap. The controller returned `ALLOCATION_FAILED`; device, daewr version and installed help usage were not observed. K7 did not execute either call. Planned calls if installed help confirms these arguments: `daewr::DefScreen(m=6, c=0, center=0, randomize=FALSE)` and likewise for `m=7`.

| k | daewr shape | TorchOED shape | Result |
|---|---|---|---|
| 6 | unavailable | 13 × 6 | NOT_RUN: ALLOCATION_FAILED |
| 7 | unavailable | 17 × 7 | NOT_RUN: ALLOCATION_FAILED |

No reference CSV or session information was produced. `run.py` checks signed column permutations against row multisets; local checks used synthetic fixtures only (two matches, one exhaustive nonmatch, one shape nonmatch).

Cleanup: stop and fresh authenticated usage passed, 0 → 0 assignments and 51.32 → 51.32 compute units. See [runtime.json](runtime.json), [cleanup.json](cleanup.json) and the [Issue blocker](https://github.com/JohnnyZPiggy/TorchOED/issues/266#issuecomment-6954710746). No retry.
