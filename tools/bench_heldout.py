"""Frozen W3 inputs and one untuned CPU held-out run; freeze before run."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import csv, json, math, platform, statistics, subprocess, sys, time
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from torchoed import full_factorial, polynomial_features, effects_coding, optimal_design, d_optimality
OUT = ROOT/'reports/w3-heldout'
CASES = {'H1':(15,'DAI'),'H2':(16,'DI'),'H3':(10,'DAI'),'H4':(24,'D')}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    I = full_factorial((3,3,3,3)); C = I[:,:3].double()-1
    H2 = torch.cat((C,effects_coding(I[:,3],3),torch.stack([C[:,i]*C[:,j] for i,j in combinations(range(3),2)],1),C.square()),1)
    I = full_factorial((11,11))-5
    H3 = I[(I.sum(1)<=5)&(I[:,0]-I[:,1]>=-7)].double()/5
    C = torch.rand(1000,5,dtype=torch.float64,generator=torch.Generator().manual_seed(20261101))*2-1
    H4 = torch.cat((C,torch.stack([C[:,i]*C[:,j] for i,j in combinations(range(5),2)],1)),1)
    matrices = [polynomial_features(full_factorial((5,5,5)).double()/2-1,2)[:,1:],H2,polynomial_features(H3,2)[:,1:],H4]
    for name,F in zip(CASES,matrices):
        with (OUT/(name+'_F.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['f'+str(i+1) for i in range(F.shape[1])])
            writer.writerows([[format(x,'.17g') for x in row] for row in F.tolist()])


def load(name):
    with (OUT/(name+'_F.csv')).open(encoding='utf-8') as f:
        reader=csv.reader(f); next(reader)
        X=torch.tensor([[float(x) for x in row] for row in reader],dtype=torch.float64)
    return torch.cat((torch.ones(len(X),1,dtype=X.dtype),X),1)


def metric(F,ids):
    X=F[ids]; M=X.mT@X
    try:
        logdet=d_optimality(M).item()
    except ValueError as error:
        return dict(D=None,A=None,I=None,error=str(error))
    inverse=torch.linalg.inv(M)
    return dict(D=logdet,A=torch.trace(inverse).item(),I=torch.trace(inverse@(F.mT@F/len(F))).item())


def summary(data):
    lines=['# W3 held-out: one run, reported as-is','',json.dumps(data['environment']),
           'D = logdet(M); A/I = trace(M^-1 B). Cells: shared value / efficiency vs best found / median seconds (R quality only). D efficiency = exp((value-best)/p); A/I = best/value. PyDOE D-selected designs are also rescored for A/I.',
           'AlgDesign: '+data['algdesign_status']+'; Julia NOT_RUN. R timing is never compared. Frozen-input oracles: '+str(data['input_oracles']),
           '', '| Case / criterion | TorchOED value / eff / s | certified floor / stop | sequential | fedorov | modified_fedorov | detmax | AlgDesign value / eff |','|---|---|---|---|---|---|---|---|']
    for name,(_,criteria) in CASES.items():
        p=load(name).shape[1]
        for c in criteria:
            rows=[r for r in data['rows'] if r['case']==name and (r['criterion']==c or r['method'].startswith('PyDOE/'))]
            values=[r['metrics'][c] for r in rows if r['metrics'][c] is not None]
            best=(max(values) if c=='D' else min(values)) if values else None
            cells=[]
            for method in ('TorchOED','PyDOE/sequential','PyDOE/fedorov','PyDOE/modified_fedorov','PyDOE/detmax','AlgDesign'):
                row=next((r for r in rows if r['method']==method),None)
                if row is None:
                    cells.append('NOT_APPLICABLE' if method.startswith('PyDOE') else 'NOT_RUN'); continue
                value=row['metrics'][c]
                eff=(math.exp((value-best)/p) if c=='D' else best/value) if value is not None else None
                cells.append('INVALID' if value is None else f'{value:.9g} / {eff:.9g}'+(f" / {row['seconds']:.6g}" if method!='AlgDesign' else ''))
            tor=next(r for r in rows if r['method']=='TorchOED')
            lines.append(f"| {name} / {c} | {cells[0]} | {tor['floor']:.9g} / {tor['stop']} | "+' | '.join(cells[1:])+' |')
    lines+=['','R D/A conversions (1e-9 relative), I ratio (conjecture only): '+json.dumps(data.get('r_conversion',[])),data.get('cleanup','No remote session allocated; controller status recorded separately.')]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def rescore_r(data):
    path=OUT/'algdesign_out.json'
    if not path.exists():
        return
    data['algdesign_status']='RUN (quality only)'; data['r_conversion']=[]
    for row in json.loads(path.read_text()):
        name,c=row['case'],row['criterion']; F=load(name); ids=[i-1 for i in row['rows']]
        scores=metric(F,ids); n,p=len(ids),F.shape[1]
        data['rows'].append(dict(case=name,criterion=c,method='AlgDesign',ids=ids,metrics=scores))
        expected=dict(D=math.exp((scores['D']-p*math.log(n))/p),A=n*scores['A']/p,I=n*scores['I'])
        data['r_conversion'].append(dict(case=name,criterion=c,DA={k:('NOT_REPORTED' if row[k] is None else 'PASS' if math.isclose(expected[k],row[k],rel_tol=1e-9) else 'FAIL') for k in ('D','A')},I_ratio=None if row['I'] is None else expected['I']/row['I']))


def run():
    import pydoe
    torch.set_num_threads(1)
    H3=torch.tensor([[i/5,j/5] for i in range(-5,6) for j in range(-5,6) if i+j<=5 and i-j>=-7],dtype=torch.float64)
    H4=torch.rand(1000,5,dtype=torch.float64,generator=torch.Generator().manual_seed(20261101))*2-1
    assert len(H3)==100 and torch.equal(H3,load('H3')[:,1:3])
    assert torch.equal(H4,load('H4')[:,1:6])
    data=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),environment=dict(cpu=platform.processor(),os=platform.platform(),python=platform.python_version(),torch=torch.__version__,pydoe=version('pydoe'),numpy=version('numpy'),scipy=version('scipy'),threads=1),input_oracles=dict(H3='PASS m=100, identical coordinates',H4='PASS torch.equal float64 round-trip'),algdesign_status='NOT_RUN',rows=[])
    assert version('pydoe')=='1.6.0'
    for name,(n,criteria) in CASES.items():
        F=load(name)
        for c in criteria:
            times=[]
            for repeat in range(3):
                begin=time.perf_counter(); result=optimal_design(F,n,criterion=c,generator=torch.Generator().manual_seed(0)); times.append(time.perf_counter()-begin)
            ids=result['ids'].tolist()
            data['rows'].append(dict(case=name,criterion=c,method='TorchOED',ids=ids,metrics=metric(F,ids),floor=result['efficiency_lower_bound'],stop=result['stop_reason'],seconds=statistics.median(times)))
            print(name,c,'TorchOED completed',flush=True)
        if name in ('H1','H3'):
            C=F[:,1:4 if name=='H1' else 3]; lookup={tuple(x):i for i,x in enumerate(C.tolist())}
            for method in ('sequential','fedorov','modified_fedorov','detmax'):
                times=[]
                for repeat in range(3):
                    begin=time.perf_counter(); result=pydoe.optimal_design(C.numpy(),n,degree=2,criterion='D',method=method); times.append(time.perf_counter()-begin)
                ids=[lookup[tuple(x)] for x in result[0].tolist()]
                data['rows'].append(dict(case=name,criterion='D',method='PyDOE/'+method,ids=ids,metrics=metric(F,ids),repeats=len(ids)-len(set(ids)),seconds=statistics.median(times)))
                print(name,method,'completed',flush=True)
        (OUT/'results.json').write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8')
    rescore_r(data)
    (OUT/'results.json').write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8'); summary(data)


if __name__=='__main__':
    if sys.argv[1]=='freeze': freeze()
    elif sys.argv[1]=='run': run()
    else:
        data=json.loads((OUT/'results.json').read_text()); rescore_r(data)
        (OUT/'results.json').write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8'); summary(data)
