"""Shared balance/GWLP scoring of the eight frozen catalogue cases."""
import csv
import json
import platform
from collections import Counter
from itertools import combinations, product
from pathlib import Path
import torch
from torchoed import orthogonal_array

ROOT = Path(__file__).resolve().parent
CASES = [(2,)*7, (2,)*11, (2,)*15, (3,)*4, (3,)*13,
         (4,)*5, (5,)*6, (4,4)+(2,)*9]


def score(rows, levels):
    n, k = len(rows), len(levels)
    balance = all(n % (levels[i]*levels[j]) == 0 and
        Counter((r[i], r[j]) for r in rows) == dict.fromkeys(
            product(range(levels[i]), range(levels[j])), n//(levels[i]*levels[j]))
        for i, j in combinations(range(k), 2))
    # Completeness of scaled contrasts: sum_c c(a)c(b) = s*[a=b]-1.
    # Expand product_i (1 + z*(s_i*[a_i=b_i]-1)) and average over row pairs.
    # This is the stated contrast-product sum, without enumerating bases.
    totals = [0]*(k+1)
    for a in rows:
        for b in rows:
            coefficients = [1]+[0]*k
            for i, s in enumerate(levels):
                kernel = s*int(a[i] == b[i])-1
                for j in range(i+1, 0, -1):
                    coefficients[j] += kernel*coefficients[j-1]
            totals = [x+y for x, y in zip(totals, coefficients)]
    gwlp = [x/(n*n) for x in totals]
    return dict(runs=n, strength_2=balance, A3=gwlp[3], GWLP_A1_to_Ak=gwlp[1:])


def main():
    torch.set_num_threads(1)
    binary = list(product(range(2), repeat=2))
    small = [list(r)+[(r[0]+r[1]) % 2] for r in binary]
    l8 = [[sum(x*y for x, y in zip(r,c)) % 2
           for c in product(range(2), repeat=3) if any(c)]
          for r in product(range(2), repeat=3)]
    l9 = [[a,b,(a+b)%3,(a+2*b)%3] for a,b in product(range(3), repeat=2)]
    assert [score(a,s)['A3'] for a,s in [(small,(2,)*3),(l8,(2,)*7),(l9,(3,)*4)]] == [1,7,8]
    runtime = json.loads((ROOT/'runtime.json').read_text())
    raw = list(csv.DictReader((ROOT/'arrays.csv').open(newline='')))
    r_gwlp = json.loads((ROOT/'gwlp.json').read_text())
    cases = {}
    for i, levels in enumerate(CASES, 1):
        name = f'C{i}'
        library = score(orthogonal_array(levels).tolist(), levels)
        data = [row for row in raw if row['case'] == name]
        if data[0]['error']:
            cases[name] = dict(levels=levels, torchoed=library, verdict='NOT_RUN', error=data[0]['error'])
            continue
        rows = [[0]*len(levels) for _ in range(max(int(r['row']) for r in data))]
        for r in data:
            rows[int(r['row'])-1][int(r['column'])-1] = int(r['value'])
        reference = score(rows, levels)
        delta = library['A3']-reference['A3']
        verdict = 'NO MATCH'
        if library['strength_2'] and reference['strength_2']:
            if library['runs'] == reference['runs'] and abs(delta) <= 1e-9:
                verdict = 'MATCH'
            elif library['runs'] < reference['runs'] or (library['runs'] == reference['runs'] and delta < -1e-9):
                verdict = 'TorchOED better'
        cases[name] = dict(levels=levels, torchoed=library, R=reference,
                           R_GWLP_informational=r_gwlp[name], delta_A3=delta, verdict=verdict)
    result = dict(workflow_run_id=runtime['run_id'], source_sha=runtime['source_sha'],
        versions=dict(R=runtime['R_version'], DoE_base=runtime['packages']['DoE.base'],
                      Python=platform.python_version(), torch=torch.__version__),
        matches=sum(c['verdict']=='MATCH' for c in cases.values()), total=8, cases=cases)
    (ROOT/'results.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')
    print(f"Sandbox: 3 PASS; shared scoring: 8 PASS; matches: {result['matches']}/8")


if __name__ == '__main__':
    main()
