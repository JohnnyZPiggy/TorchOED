"""Re-score the frozen Week 10 references on CPU float64."""
import csv
import re
from pathlib import Path

import torch

from torchoed import fractional_factorial, word_length_pattern

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def group(name, key, value, metadata, convert):
    return [[convert(cell) for column, cell in row.items() if column not in metadata and cell]
            for row in read(name) if row[key] == str(value)]


def main():
    torch.set_num_threads(1)
    catlg = ROOT / "frf2_catlg.csv"
    if catlg.exists():
        help_text = (ROOT / "sessionInfo.txt").read_text(encoding="utf-8")
        documented = re.search(r"starting with words of length\s+(\d+)", help_text)
        assert documented, "The installed FrF2 help must establish the WLP start index"
        j0 = int(documented[1])
        expected = {f"{k}-{k-m}.1" for m in range(3, 7)
                    for k in range(m + 1, min(11, 2**m - 1) + 1)}
        rows = read(catlg.name)
        matches = 0
        for row in rows:
            n, k = int(row["runs"]), int(row["k"])
            m = n.bit_length() - 1
            gen = [[c for c in range(m) if int(g) & (1 << c)] for g in row["gen"].split(";")]
            x = fractional_factorial(m, gen, dtype=torch.float64, device="cpu")
            assert x.shape == (n, k), row["name"]
            wlp = word_length_pattern(x)
            scaled = wlp * n**2
            assert torch.equal(scaled, scaled.round()), row["name"]
            actual = wlp.round().to(torch.int64).tolist()
            observed = [int(v) for v in row["WLP"].split(";")] if row["WLP"] else []
            aligned = [0] * (j0 - 1) + observed
            aligned += [0] * (k - len(aligned))
            if not wlp[:j0 - 1].any() and actual == aligned:
                matches += 1
            else:
                print(f'{row["name"]} mismatch: TorchOED {actual}, FrF2 (j0={j0}) {observed}')
            if row["name"] == "7-4.1":
                print(f"7-4.1 stored WLP: {observed}, j0={j0}")
        print(f"Catalogue: {len(rows)}/{len(expected)} rows, {matches}/{len(rows)} WLP matches")
        print(f'Missing names: {sorted(expected - {row["name"] for row in rows})}')
    else:
        print("Catalogue WLP: NOT_RUN")

    if (ROOT / "frf2_pb.csv").exists():
        for n in (12, 20, 24):
            x = torch.tensor(group("frf2_pb.csv", "N", n, {"N"}, int), dtype=torch.int64)
            assert x.shape == (n, n - 1) and (x.abs() == 1).all(), n
            assert torch.equal(x.T @ x, n * torch.eye(n - 1, dtype=torch.int64)), n
            print(f"PB {n}: exact Gram PASS")
    else:
        print("PB: NOT_RUN")

    if (ROOT / "rsm_ccd.csv").exists():
        for case, n, k in (("A", 14, 2), ("B", 30, 4), ("C", 46, 5)):
            file = "rsm_ccd.csv"
            x = torch.tensor(group(file, "case", case, {"case", "block"}, float), dtype=torch.float64)
            assert x.shape == (n, k), case
            blocks = [r["block"] for r in read(file) if r["case"] == case]
            axial = x[torch.tensor([b == blocks[-1] for b in blocks], dtype=torch.bool)]
            print(f"CCD {case}: shape {n}x{k}, alpha={axial.abs().max().item():.15g}")
    else:
        print("CCD: NOT_RUN")
    if (ROOT / "rsm_bbd.csv").exists():
        for k, n in ((3, 15), (5, 43)):
            rows = group("rsm_bbd.csv", "k", k, {"k"}, float)
            assert len(rows) == n and all(len(row) == k for row in rows), k
            print(f"BBD {k}: shape {n}x{k} PASS")
    else:
        print("BBD: NOT_RUN")


if __name__ == "__main__":
    main()
