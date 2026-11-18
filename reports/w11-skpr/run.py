"""Local structure check only; TorchOED re-scoring NOT_RUN pending D2 gate."""
import csv
from pathlib import Path

p = Path(__file__).with_name('skpr.csv')
if not p.exists():
    print('NOT_RUN')
else:
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 24
    assert all(float(r['W']) in (-1,0,1) and all(float(r[k]) in (-1,1) for k in ('X1','X2')) for r in rows)
    plots = {r['rowname'].rsplit('.',1)[0] for r in rows}
    assert all(len({r['W'] for r in rows if r['rowname'].rsplit('.',1)[0] == plot}) == 1 for plot in plots)
    print('PASS: 24 rows; levels; W constant on',len(plots),'whole-plot prefixes')
