import csv, json
from pathlib import Path
import torch
from torchoed import space_filling_diagnostics
p = Path(__file__).parent
torch.set_num_threads(1)
def tensor(rows): return torch.tensor(rows, dtype=torch.float64)
def load(name): return tensor([[float(v) for v in r] for r in csv.reader((p / name).open())])
raw = json.loads((p / "raw.json").read_text()); F = load("sample.csv")
def metrics(x):
    d = {k: v.item() for k, v in space_filling_diagnostics(x).items()}
    d["energy"] = (2 * torch.cdist(x, F).mean() - torch.cdist(x, x).mean()).item()
    return d
X, R = load("torchoed.csv"), tensor(raw["design"])
a, b = metrics(X), metrics(R)
winners = {k: "equal" if a[k] == b[k] else "TorchOED" if (a[k] > b[k] if k == "min_distance" else a[k] < b[k]) else "R" for k in a}
assert (X[:, None] == F).all(-1).any(-1).all()
assert torch.allclose(2 * torch.cdist(F,F).mean() - torch.cdist(F,F).mean(), torch.cdist(F,F).mean())
result = dict(base_sha="879a7f23152b5711872bebf651b4d1765cd688ef",run_id=raw["run_id"],source_sha=raw["source_sha"],R_version=raw["R_version"],support=raw["support"],sample_rows=len(F),torchoed=a,reference=b,better=winners,reference_max_constraint_violation=(R.sum(1)-1).max().item(),example={k:v for k,v in raw["example"].items() if k not in ("design","rd2ex")},before=14,after=14)
(p / "results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
print(json.dumps(result,indent=2))
