"""Score the frozen W10 held-out cases on CPU in float64."""
import csv
import json
import platform
from collections import Counter
from functools import reduce
from itertools import combinations
from operator import xor
from pathlib import Path

import torch

from torchoed import (
    box_behnken, central_composite, fractional_factorial,
    fractional_factorial_generators, full_factorial, polynomial_features,
    word_length_pattern,
)

OUT = Path(__file__).resolve().parent


def h21(rows, generators):
    wlp = word_length_pattern(rows).tolist()
    words = [sum(1 << c for c in term) | (1 << i)
             for i, term in enumerate(generators, start=4)]
    counts = Counter(reduce(xor, terms).bit_count()
                     for n in range(1, 4) for terms in combinations(words, n))
    passed = wlp == [counts[j] for j in range(1, 8)]
    return dict(A3_A7=wlp[2:7], resolution=next(j for j, a in enumerate(wlp, 1) if a),
                generators=generators, oracle="PASS" if passed else "FAIL")


def h22(rows, blocks):
    f = polynomial_features(rows, 2, include_bias=False)
    z = (blocks[:, None] == blocks.unique()).to(f)
    centered = torch.cat([f[b] - f[b].mean(0) for b in z.T.bool()])
    mb = centered.T @ centered
    projector = torch.eye(len(f), dtype=f.dtype) - z @ torch.linalg.solve(z.T @ z, z.T)
    passed = torch.allclose(mb, f.T @ projector @ f, rtol=0, atol=1e-10)
    fc = f - f.mean(0)
    return dict(D_B=float(torch.linalg.det(mb / len(f)).pow(1 / 9)),
                max_centered_gap=float((fc.T @ fc - mb).abs().max()),
                alpha=float(rows[z[:, 1].bool()].abs().max()),
                oracle="PASS" if passed else "FAIL")


def h23(rows):
    f = polynomial_features(rows, 2)
    g = polynomial_features(full_factorial([3] * 4).double() - 1, 2)
    m = f.T @ f / len(f)
    w = g.T @ g / len(g)
    rank = int(torch.linalg.matrix_rank(f))
    i = torch.trace(w @ torch.linalg.inv(m))
    oracle = torch.trace(torch.linalg.solve(m, w))
    passed = rank == 15 and torch.isclose(i, oracle, rtol=1e-10, atol=0)
    return dict(D=float(torch.linalg.det(m).pow(1 / 15)), I=float(i), rank=rank,
                oracle="PASS" if passed else "FAIL")


def reference(case):
    with (OUT / f"{case}.csv").open(newline="") as file:
        values = list(csv.reader(file))[1:]
    rows = torch.tensor([[float(x) for x in row] for row in values], dtype=torch.float64)
    if case == "H21":
        # Identify each generated column's signed monomial on the four base columns.
        terms = [term for n in range(1, 5) for term in combinations(range(4), n)]
        generators = [next(term for term in terms if
                           (rows[:, term].prod(1) * column).sum().abs() == len(rows))
                      for column in rows[:, 4:].T]
        return h21(rows, generators)
    if case == "H22":
        return h22(rows[:, :3], rows[:, 3])
    return h23(rows)


if __name__ == "__main__":
    torch.set_num_threads(1)
    runtime = json.loads((OUT / "runtime.json").read_text(encoding="utf-8"))
    g = fractional_factorial_generators(7, 16)
    locals_ = (h21(fractional_factorial(4, g), g),
               h22(central_composite(3, alpha="orthogonal", center=(4, 2)),
                   torch.tensor([1] * 12 + [2] * 8)),
               h23(box_behnken(4, center=3)))
    results = dict(source_sha=runtime["source_sha"], python=platform.python_version(),
                   torch=torch.__version__, device="cpu", dtype="float64", threads=1, cases={})
    for case, local in zip(("H21", "H22", "H23"), locals_):
        local["status"] = local["oracle"]
        error = runtime["reference_errors"].get(case)
        if error is None:
            ref = reference(case)
            ref["status"] = ref["oracle"]
        else:
            ref = dict(status="NOT_RUN", oracle="NOT_RUN", error=error)
        results["cases"][case] = dict(TorchOED=local, reference=ref)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8", newline="\n")
