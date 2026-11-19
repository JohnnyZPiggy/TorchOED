import csv
from pathlib import Path
import torch
from torchoed import lhs
p = Path(__file__).parent
torch.set_num_threads(1)
for mode in ("random", "maximin", "maxpro", "uniform"):
    X = lhs(8, 3, mode=mode, generator=torch.Generator().manual_seed(0))
    with (p / (mode + ".csv")).open("w", newline="") as f:
        csv.writer(f).writerows([[repr(v) for v in row] for row in X.tolist()])
