"""Re-score frozen Xvert outputs and an independent exact vertex enumeration."""
import csv
import json
import platform
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
import torch
from torchoed import mixture_vertices

ROOT = Path(__file__).resolve().parent
CONVENTION = ('Xvert exposes nlc/lb/ub/coef, not conmx; installed source builds '
              'conmx rows (a,b) meaning a*x+b >= 0 and drops b=0 rows. '
              'R1: .2 <= x1+x2 <= .75. R2: 1 <= 2*x1+x3+x4 <= 2, '
              'equivalent to x1-x2 >= 0 on sum(x)=1 with the box bounds.')


def exact_vertices(lower, upper, constraint):
    q = len(lower)
    inequalities = []  # a*x <= b, including both sides of the box.
    for j in range(q):
        e = [Q(i == j) for i in range(q)]
        inequalities.extend([(e, Q(upper[j])), ([-x for x in e], -Q(lower[j]))])
    a, lo, hi = constraint
    if lo is not None:
        inequalities.append(([-Q(x) for x in a], -Q(lo)))
    if hi is not None:
        inequalities.append(([Q(x) for x in a], Q(hi)))
    vertices = set()
    for active in combinations(inequalities, q-1):
        rows = [[Q(1)] * (q+1)] + [list(a)+[b] for a, b in active]
        for j in range(q):
            pivot = next((i for i in range(j, q) if rows[i][j]), None)
            if pivot is None:
                break
            rows[j], rows[pivot] = rows[pivot], rows[j]
            scale = rows[j][j]
            rows[j] = [x/scale for x in rows[j]]
            for i in range(q):
                if i != j:
                    scale = rows[i][j]
                    rows[i] = [x-scale*y for x, y in zip(rows[i], rows[j])]
        else:
            v = tuple(row[-1] for row in rows)
            if all(sum(x*y for x, y in zip(a, v)) <= b for a, b in inequalities):
                vertices.add(v)
    return [[float(x) for x in row] for row in sorted(vertices)]


def score(vertices, reference):
    """Symmetric maximum of nearest-vertex absolute coordinate errors."""
    distances = (vertices[:, None, :] - torch.tensor(reference, dtype=torch.float64)[None, :, :]).abs().amax(2)
    return float(torch.maximum(distances.amin(0).amax(), distances.amin(1).amax()))


def main():
    torch.set_num_threads(1)
    runtime = json.loads((ROOT/'runtime.json').read_text())
    cases = {}
    for name, lower, upper, constraint, count in [
        ('R1', ['.1']*3, ['.7','.6','.5'], ([1,1,0], None, '.75'), 5),
        ('R2', ['.05']*4, ['.5']*4, ([1,-1,0,0], '0', None), 10)]:
        exact = exact_vertices(lower, upper, constraint)
        assert len(exact) == count
        vertices = mixture_vertices(lower, upper, constraints=[constraint], dtype=torch.float64, device='cpu')
        exact_gap = score(vertices, exact)
        text = (ROOT/f'{name}.csv').read_text()
        data = list(csv.DictReader(text.splitlines()))
        if not data or 'dimen' not in data[0]:
            cases[name] = dict(verdict='NOT_RUN', error=text, exact_count=count,
                               torchoed_count=len(vertices), exact_max_abs=exact_gap)
            continue
        rows = [r for r in data if float(r['dimen']) == 0]
        reference = [[float(r[f'x{i+1}']) for i in range(len(lower))] for r in rows]
        r_gap = score(vertices, reference)
        cases[name] = dict(torchoed_count=len(vertices), exact_count=count, xvert_count=len(rows),
                           centroids_dropped=len(data)-len(rows), exact_max_abs=exact_gap, xvert_max_abs=r_gap,
                           verdict='MATCH' if len(vertices)==len(rows)==count and exact_gap<=1e-12 and r_gap<=1e-6 else 'NO MATCH')
    result = dict(run_id=runtime['run_id'], source_sha=runtime['source_sha'],
                  versions=dict(R=runtime['R_version'], mixexp=runtime['packages']['mixexp'],
                                python=platform.python_version(), torch=torch.__version__),
                  device='cpu', dtype='float64', convention=CONVENTION, cases=cases,
                  matched=sum(c['verdict']=='MATCH' for c in cases.values()), total=2)
    (ROOT/'results.json').write_bytes((json.dumps(result, indent=2, allow_nan=False)+'\n').encode())
    print(f"PASS: exact counts 5/10 and shared scoring; MATCH {result['matched']}/2")


if __name__ == '__main__':
    main()
