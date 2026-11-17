import csv
from itertools import permutations, product
from pathlib import Path

from torchoed import definitive_screening

path = Path(__file__).with_name("daewr_dsd.csv")
if not path.exists():
    raise SystemExit("NOT_RUN: no frozen daewr CSV (see runtime.json)")
with path.open(newline="") as stream:
    rows = list(csv.reader(stream))[1:]

for k in (6, 7):
    ref = [tuple(map(int, row[1 : k + 1])) for row in rows if int(row[0]) == k]
    target = sorted(map(tuple, definitive_screening(k).to(int).tolist()))
    print(f"k={k}: daewr {(len(ref), k)}, TorchOED {(len(target), k)}")
    if len(ref) == len(target):
        for perm, signs in product(permutations(range(k)), product((-1, 1), repeat=k)):
            if sorted(tuple(s * row[i] for i, s in zip(perm, signs)) for row in ref) == target:
                print(f"MATCH (zero-based permutation={perm}, signs={signs})")
                break
        else:
            print("NO MATCH", ref, target)
    else:
        print("NO MATCH (shape)", ref, target)
