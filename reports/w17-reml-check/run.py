"""Frozen CSVs -> exact balanced EMS and one shared relative metric; no refits."""
import csv
import json
import math
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def strip_input():
    """Documentation only: the committed CSV, not this generator, is input."""
    import numpy as np
    rng = np.random.default_rng(17)
    u = rng.normal(0, 2, 4); v = rng.normal(0, 1.5, (4, 3))
    w = rng.normal(0, 1, (4, 4)); e = rng.normal(0, 1, (4, 3, 4))
    return [[h+1, i+1, j+1, f'{10+[0,1,-1][i]+[0,.5,1,-.5][j]+u[h]+v[h,i]+w[h,j]+e[h,i,j]:.4f}']
            for h, i, j in product(range(4), range(3), range(4))]


def inverse(a):
    n = len(a); a = [list(row) + [Q(i == j) for j in range(n)] for i, row in enumerate(a)]
    determinant = Q(1)
    for i in range(n):
        pivot = a[i][i]; determinant *= pivot
        a[i] = [v / pivot for v in a[i]]
        for j in range(n):
            if j != i:
                scale = a[j][i]; a[j] = [x-scale*y for x, y in zip(a[j], a[i])]
    return [row[n:] for row in a], determinant


def ems(y, b, a, c, crossed):
    n = len(y); B, A, C = map(lambda v: len(set(v)), (b, a, c))
    def means(keys):
        groups = {}
        for key, val in zip(keys, y): groups.setdefault(key, []).append(val)
        return [sum(groups[key]) / len(groups[key]) for key in keys]
    g = [sum(y)/n] * n; mb = means(b); ma = means(a); mc = means(c)
    mba = means(list(zip(b, a))); mbc = means(list(zip(b, c))); mac = means(list(zip(a, c)))
    parts = [[v-z for v,z in zip(mb,g)], [v-x-z+t for v,x,z,t in zip(mba,mb,ma,g)]]
    dfs = [(B-1), (B-1)*(A-1)]
    if crossed:
        parts += [[v-x-z+t for v,x,z,t in zip(mbc,mb,mc,g)]]; dfs += [(B-1)*(C-1)]
    resid = [val-cell-sum(part[i] for part in parts) for i,(val,cell) in enumerate(zip(y,mac))]
    parts += [resid]; dfs += [(B-1)*(A-1)*(C-1) if crossed else A*(B-1)*(C-1)]
    ms = [sum(v*v for v in part)/df for part,df in zip(parts,dfs)]
    variance = [(ms[0]-ms[1]-(ms[2]-ms[-1] if crossed else 0))/(A*C), (ms[1]-ms[-1])/C]
    if crossed: variance += [(ms[2]-ms[-1])/A]
    return ms, dfs, variance+[ms[-1]]


def exact(rows, crossed):
    bkey, akey, ckey, ykey = ('block','A','B','y') if crossed else ('Block','Variety','nitro','yield')
    b,a,c = [[row[k] for row in rows] for k in (bkey,akey,ckey)]; y=[Q(row[ykey]) for row in rows]
    ms, dfs, variance = ems(y,b,a,c,crossed)
    if min(variance) <= 0: return {'status':'N/A', 'variance':list(map(float,variance))}
    av,cv = sorted(set(a)),sorted(set(c),key=lambda v: Q(v))
    x = [[Q(1)] + [Q(i==v) for v in av[1:]] + [Q(j==v) for v in cv[1:]] +
         [Q(i==v and j==w) for w in cv[1:] for v in av[1:]] for i,j in zip(a,c)]
    n,p = len(y),len(x[0]); xtx = [[sum(row[i]*row[j] for row in x) for j in range(p)] for i in range(p)]
    inv, det = inverse(xtx); weights = [[sum(inv[i][j]*row[j] for j in range(p)) for row in x] for i in range(p)]
    fixed = [sum(w*v for w,v in zip(ws,y)) for ws in weights]
    # Orthogonal random strata: block, block:A, optional block:B, residual.
    def project(w, keys):
        sums = {}; counts = {}
        for key,val in zip(keys,w): sums[key]=sums.get(key,Q(0))+val; counts[key]=counts.get(key,0)+1
        return [sums[key]/counts[key] for key in keys]
    se, df = [], []
    for w in weights:
        pb=project(w,b); pba=project(w,list(zip(b,a))); pbc=project(w,list(zip(b,c)))
        strata=[pb,[v-z for v,z in zip(pba,pb)]]
        if crossed: strata += [[v-z for v,z in zip(pbc,pb)]]
        strata += [[v-sum(part[i] for part in strata) for i,v in enumerate(w)]]
        terms=[sum(v*v for v in part)*lam for part,lam in zip(strata,ms)]
        se.append(math.sqrt(sum(terms))); df.append(float(sum(terms)**2/sum(t*t/d for t,d in zip(terms,dfs))))
    ll=-.5*((n-p)*(math.log(2*math.pi)+1)+math.log(det)+sum(d*math.log(lam) for d,lam in zip(dfs,ms)))
    return dict(fixed=list(map(float,fixed)),SE=se,df=df,variance=list(map(float,variance)),logLik=[ll])


def score(a,b):
    return max(abs(x-y)/max(abs(y),1e-300) for x,y in zip(a,b)) if len(a)==len(b) else None


def main():
    # Exact non-data check: 2 blocks x 2 plots x 2 subplots, known orthogonal strata.
    keys=list(product(range(2),repeat=3)); y=[Q(10+4*(-1)**b+2*(-1)**(b+a)+(-1)**(b+a+c)) for b,a,c in keys]
    assert ems(y,*zip(*keys),False)[2] == [Q(24),Q(14),Q(4)]
    r={}
    for row in csv.DictReader((ROOT/'r.csv').open()): r.setdefault(row['case'],{}).setdefault(row['group'],[]).append(float(row['value']))
    cases={}
    for name in ('Oats','strip'):
        ex=exact(list(csv.DictReader((ROOT/f'{name}.csv').open())),name=='strip')
        # R VarCorr orders nested before parent for Oats; crossed follows formula order.
        order = [1,0,2] if name=='Oats' else [2,1,0,3]
        r[name]['variance']=[r[name]['variance'][i] for i in order]
        cases[name]={'R':r[name],'exact':ex,'PyStatistics':'NOT_RUN: install failed (pystatistics.txt)',
                     'R_vs_exact':{g:score(r[name][g],ex[g]) for g in r[name] if ex.get('status')!='N/A'},
                     'PyStatistics_vs_exact':'NOT_RUN','PyStatistics_vs_R':'NOT_RUN'}
    result={'decision':'BUILD','PyStatistics_version':'7.0.0','Python':'3.14.6','R_run':45035411313,
            'pins':{'R':'4.5.2','lme4':'2.1-0','lmerTest':'3.2-1'},'cases':cases}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print('EMS self-check PASS; frozen results reproduced; decision BUILD')


if __name__=='__main__': main()
