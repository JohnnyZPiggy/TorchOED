import csv, importlib.util, json, math, platform, time
from pathlib import Path
import torch
from torchoed import fit_reml, fit_least_squares, predict_least_squares, d_max, d_min, d_overall, maximize_desirability
p=Path(__file__).resolve().parent; torch.set_num_threads(1); start=time.perf_counter()
spec=importlib.util.spec_from_file_location('ems',p.parent/'w17-reml-check/run.py'); em=importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
def read(name): return json.loads((p/name).read_text(encoding='utf-8-sig'))
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def tensor(x): return torch.tensor(x,dtype=torch.float64)
def group(x): return torch.tensor([sorted(set(x)).index(v) for v in x])
old=json.loads((p.parent/'w17-reml-check/results.json').read_text())['cases']['Oats']['exact']
reg=em.exact(rows(p.parent/'w17-reml-check/Oats.csv'),False)
assert all(em.score(reg[k],old[k])<=1e-12 for k in old)
rt=read('runtime.json'); ref=read('r.json'); cases={}
versions_ok=all(rt['packages'].get(k)==v for k,v in [('lme4','2.1-0'),('lmerTest','3.2-1')]) and '4.5.2' in rt['R_version']
if 'torchoed' not in rt:
    ours={}; r=ref['H49']
    if 'error' not in r:
        cake=rows(p/'cake.csv'); X=tensor(r['X']); y=tensor([float(v['angle']) for v in cake])
        mapped=[dict(Block=v['replicate'],Variety=v['recipe'],nitro=v['temp'],**{'yield':v['angle']}) for v in cake]
        a=sorted(set(v['recipe'] for v in cake)); c=sorted(set(v['temp'] for v in cake),key=float)
        expected=tensor([[1]+[v['recipe']==i for i in a[1:]]+[v['temp']==j for j in c[1:]]+[v['recipe']==i and v['temp']==j for j in c[1:] for i in a[1:]] for v in cake])
        assert torch.equal(X,expected)
        fit=fit_reml(X,y,(group([v['replicate'] for v in cake]),group([v['replicate']+':'+v['recipe'] for v in cake])))
        ours['H49']={k:fit[v].reshape(-1).tolist() for k,v in [('fixed','coef'),('SE','se'),('df','df'),('variance','variance'),('logLik','loglik')]}
        ours['H49_exact']=em.exact(mapped,False)
    if 'error' not in ref['H50']:
        sl=rows(p/'stackloss.csv'); X=tensor([[1]+[float(v[k]) for k in ('Air.Flow','Water.Temp','Acid.Conc.')] for v in sl]); y=tensor([float(v['stack.loss']) for v in sl])
        pred=predict_least_squares(fit_least_squares(X,y),tensor([[1,60,21,87]]))
        ours['H50']=[float(pred['fit'][0]),float(pred['se'][0]),*pred['ci'][0].tolist(),*pred['pi'][0].tolist()]
    def overall(x):
        a,b=x.unbind(1)
        return d_overall(d_max(80+5*a+3*b,70,90),d_min(10-2*a+2*b,5,20))
    best=maximize_desirability(overall,tensor([-1,-1]),tensor([1,1]))
    ours['H51']=dict(point=best['x'].tolist(),D=float(best['value']))
    rt.update(torchoed=ours,python=platform.python_version(),torch=torch.__version__,threads=torch.get_num_threads(),wall_seconds_torchoed=time.perf_counter()-start)
    (p/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8',newline='\n')
ours=rt['torchoed']
for name in ('H49','H50'):
    r=ref[name]
    if 'error' in r or not versions_ok:
        cases[name]=dict(verdict='NOT_RUN',reference='NOT_RUN',error=r.get('error','Reference versions not recorded at required pins')); continue
    if name=='H49':
        r={k:([r[k]] if isinstance(r[k],(int,float)) else r[k]) for k in ('fixed','SE','df','variance','logLik')}; ex=ours['H49_exact']; actual=ours[name]
        rd={k:em.score(actual[k],r[k]) for k in r}; boundary=ex.get('status')=='N/A'
        ed={} if boundary else {k:em.score(actual[k],ex[k]) for k in r}
        worst=max(rd if boundary else ed,key=(rd if boundary else ed).get)
        match=all(v<=(1e-8 if k in ('fixed','logLik') else 1e-4) for k,v in rd.items()) if boundary else max(ed.values())<=1e-10
        cases[name]=dict(torchoed=actual,R=r,exact='NOT_RUN (boundary)' if boundary else ex,exact_variance=ex['variance'],vs_R=rd,vs_exact=ed,R_vs_exact={} if boundary else {k:em.score(r[k],ex[k]) for k in r},worst_quantity=worst,verdict='MATCH' if match else 'NO MATCH',column_order=ref[name]['columns'])
    else:
        cf=r['confidence']; cfv=cf['fit'][0] if isinstance(cf['fit'][0],list) else cf['fit']; pv=r['prediction'][0] if isinstance(r['prediction'][0],list) else r['prediction']
        se=cf['se.fit']; expected=[cfv[0],se[0] if isinstance(se,list) else se,*cfv[1:],*pv[1:]]
        dif=[abs(a-b)/max(abs(b),1e-300) for a,b in zip(ours[name],expected)]; names=['fit','se','ci.lwr','ci.upr','pi.lwr','pi.upr']
        cases[name]=dict(torchoed=ours[name],reference=expected,max_relative=max(dif),worst_quantity=names[dif.index(max(dif))],verdict='MATCH' if max(dif)<=1e-8 else 'NO MATCH')
a=ours['H51']; delta=abs(a['D']-math.sqrt(.605)); point=max(abs(x-y) for x,y in zip(a['point'],[1,.5]))
cases['H51']=dict(torchoed=a,reference=dict(point=[1,.5],D=math.sqrt(.605)),D_absolute=delta,point_max_absolute=point,verdict='MATCH' if delta<=1e-8 and point<=1e-5 else 'NO MATCH')
result=dict(run_id=rt['run_id'],source_sha=rt['source_sha'],library_sha='94361f75319744ccaf2ebc716d08feeb2e53b9b7',versions=dict(R=rt['R_version'],lme4=rt['packages'].get('lme4'),lmerTest=rt['packages'].get('lmerTest'),Python=rt['python'],torch=rt['torch']),checks=dict(Oats_exact_regression='PASS at 1e-12',column_order='PASS: treatment contrasts, first factor fastest'),cases=cases)
(p/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print({k:v['verdict'] for k,v in cases.items()},'Oats regression PASS; seconds',time.perf_counter()-start)
