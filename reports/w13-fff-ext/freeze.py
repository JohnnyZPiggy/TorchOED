import csv
from pathlib import Path
import torch
from torchoed import sobol, feasible_candidates, fast_flexible_filling
p = Path(__file__).parent
torch.set_num_threads(1)
A, b = torch.tensor([[1., 1.]], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)
S = sobol(1024, 2, scramble=True, seed=0); F = S[feasible_candidates(S, A=A, b=b)]
X = fast_flexible_filling(10, 2, A=A, b=b, n_candidates=1024, seed=0)
assert (X[:, None] == F).all(-1).any(-1).all()
for name, x in (("sample", F), ("torchoed", X)):
    with (p / (name + ".csv")).open("w", newline="") as f: csv.writer(f).writerows(x.tolist())
print(f"Freeze PASS: {len(F)} feasible rows; all 10 design rows in sample")
