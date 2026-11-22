# Week 16 held-out H43–H45

All three frozen cases MATCH on base dd7a96486f3db92eb89ee1baa0e675708219fec5 (#524).
TorchOED calls ran once with seed 0 where prescribed and default settings. runtime.json preserves their outputs and wall times. Later run.py invocations only re-score saved outputs and references, without rerunning search or Monte Carlo.

| Case | TorchOED | Reference | Verdict |
|---|---|---|---|
| H43 blocked MA (10, 64, 4) | WLP [0,0,0,2,8,4,0,1,0,0]; no main/2FI block confounding | Same WLP; no main/2FI block confounding | MATCH |
| H44 Poisson, 10,000 simulations | [0.5107, 0.7336, 0.2561, 0.0441] | [0.5230, 0.7322, 0.2559, 0.0465] | MATCH; worst z = 1.7406031030 < 4 |
| H45 strip A (2 x 4), eta_row=2, eta_col=1/2 | Exact trace 13/4 | Exhaustive minimum 13/4; 960 nonsingular feasible assignments, 48 minimizers of 16,384 total | MATCH; delta 0 |

R oracle: [run 45009364423](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45009364423), source 3f1c4531c88fb4215acdad609551b0cf4d7a2785; R 4.5.2, FrF2 2.3-5, skpr 1.9.2. H44 CSV was frozen in prior commit eefdd79. One dispatch, no reference failures or changed-call retries. Full sessionInfo() and downloaded outputs are included.

H43 confounding uses one function for both designs; its non-held-out 2² check returns only AB (zero-based [0,1]). H45 uses exact rational information matrices, scaled integer Gram matrices and a second dense rational inverse path; the traces agree exactly. Evidence uses the prescribed base without #545 changes.

Week-16 suite timing, ubuntu-latest / Python 3.13: start [44874939651](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44874939651), 388 passed in 71.64 s (pytest step wall 73 s); end [45006900758](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45006900758), 392 passed in 55.05 s (step wall 56 s), zero failures and one warning each. These are different GitHub hosts and changed suites, not a controlled speed-up claim. Job IDs and SHAs are in runtime.json.

Reproduce: PYTHONPATH=src python reports/w16-heldout/run.py. Exact results and independent metric checks only. No tuning, extra seeds, GPU or production changes.
