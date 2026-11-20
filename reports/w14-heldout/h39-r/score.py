import csv
import json
from pathlib import Path

import torch
from torchoed import space_filling_diagnostics

p = Path(__file__).parent
torch.set_num_threads(1)
X = torch.tensor([[float(v) for v in row] for row in csv.reader((p / "H39-R.csv").open())], dtype=torch.float64)
assert X.shape == (15, 3)
assert ((X >= 0) & (X <= 1)).all() and (X.sum(1) <= 1.5).all()
metrics = space_filling_diagnostics(X)
reference = {key: metrics[key].item() for key in ("maxpro", "min_distance")}
frozen = json.loads((p.parent / "results.json").read_text())["H39"]["torchoed"]
runtime = json.loads((p / "runtime.json").read_text())
delta = frozen["maxpro"] - reference["maxpro"]
verdict = "MATCH or TorchOED better" if frozen["maxpro"] <= reference["maxpro"] * (1 + 1e-9) else "NO MATCH"
result = dict(torchoed=frozen, reference=reference, verdict=verdict, delta=delta,
              run_id=runtime["run_id"], source_sha=runtime["source_sha"],
              R_version=runtime["R_version"], support=runtime["packages"]["support"])
(p / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
summary = (f"# H39 R reference completion\n\n"
           f"Run {result['run_id']} on {result['source_sha']}; {result['R_version']}; support {result['support']}.\n"
           f"MaxPro: TorchOED {frozen['maxpro']}, R {reference['maxpro']}; delta {delta}.\n"
           f"Minimum distance: TorchOED {frozen['min_distance']}, R {reference['min_distance']}.\n"
           f"Verdict: {verdict}; frozen relative tolerance 1e-9.\n"
           "R feasibility PASS (15 rows in [0,1]^3, sums <= 1.5); TorchOED values read unchanged from ../results.json.\n")
(p / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
print(json.dumps(result, indent=2))
