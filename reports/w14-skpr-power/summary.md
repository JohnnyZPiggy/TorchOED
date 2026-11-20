# W14 skpr power across run sizes

R oracle run [44690838539](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44690838539), source `794275a1c916bb2af20fe541d3ec3fb767b36153`.
R version 4.5.2 (2025-10-31); skpr 1.9.2. TorchOED CPU float64, Python 3.14.6, torch 2.14.1+cpu.

| Design | df | max absolute power delta | Verdict |
|---|---|---|---|
| D8 | 1 | 2.0730739347385452e-10 | NO MATCH |
| D12 | 5 | 1.3812218035980095e-10 | NO MATCH |
| D16 | 9 | 1.9084478442010777e-10 | NO MATCH |

Mapping fixed before dispatch: skpr effectsize=2 means coefficient=1 for continuous terms; TorchOED effect=1, alpha=0.05. Six non-intercept parameter powers per design compared. Intercept excluded; its skpr number is retained as-is in results.json and reference.json.

NR: optional Rd run omitted; only the three frozen required designs evaluated; count remains 14. No published-example index increment.

Reproduce: `python reports/w14-skpr-power/run.py`. No searches, held-out cases, GPU, or production changes.
