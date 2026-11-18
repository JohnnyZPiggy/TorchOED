"""#339 frozen replicated covariate case; --score never repeats the search/oracle."""
import csv, itertools, json, math, platform, subprocess, sys
from fractions import Fraction
from pathlib import Path
import torch
from torchoed import optimal_design
torch.set_num_threads(1); path=Path(__file__).with_name('results.json'); points=list(itertools.product((-1,0,1),(-1,1))); rows=[(h,x,h*x) for x,h in points]
F=torch.tensor(rows*5,dtype=torch.float64); Z=torch.arange(5,dtype=F.dtype)[:,None]; X=torch.cat((torch.ones_like(Z),Z),1); Q=torch.eye(5,dtype=F.dtype)-X@torch.linalg.solve(X.T@X,X.T)
def score(rows): return torch.linalg.det((s:=torch.tensor(rows,dtype=F.dtype)).T@Q@s).item()
def record(rows):
    value=score(rows); return dict(rows=rows,det_M_Z=value,verdict='MATCH' if math.isclose(value,81.92,rel_tol=1e-10) else 'NO MATCH')
if '--score' not in sys.argv:
    r=optimal_design(F,5,criterion='D',covariates=Z,generator=torch.Generator().manual_seed(0)); result=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=platform.python_version(),torch=torch.__version__,device='CPU',dtype='float64',threads=1,seed=0,restarts=10)
    result['TorchOED']=dict(record(F[r['ids']].tolist()),ids=r['ids'].tolist(),objective=r['objective'],covariates_equal=torch.equal(r['covariates'],Z)); best=Fraction(0); ties=visited=0
    for ids in itertools.product(range(6),repeat=5):
        s=[rows[i] for i in ids]; M=[[sum(s[a][i]*(10*(a==b)-2-(a-2)*(b-2))*s[b][j] for a in range(5) for b in range(5)) for j in range(3)] for i in range(3)]
        a,b,c=M[0]; d,e,f=M[1]; g,h,i=M[2]; value=Fraction(a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g),1000); visited+=1
        if value>best: best=value; ties=0
        if value==best: ties+=1
    result['oracle']=dict(visited=visited,maximum=str(best),maximisers=ties); path.write_text(json.dumps(result,indent=2)+'\n')
result=json.loads(path.read_text()); o=result['oracle']; assert (o['visited'],o['maximum'],o['maximisers'])==(7776,'2048/25',24)
assert result['TorchOED']['covariates_equal'] and math.isclose(math.log(score(result['TorchOED']['rows'])),result['TorchOED']['objective'],rel_tol=1e-10)
result['tests']=dict(exhaustive_oracle='PASS',dense_rescore='PASS')
if (p:=path.with_name('skpr.csv')).exists():
    raw=list(csv.DictReader(p.open())); z=[float(r['Z']) for r in raw]; assert len(raw)==5 and all(float(r['h']) in (-1,1) and float(r['x']) in (-1,0,1) for r in raw)
    step=z[1]-z[0]; affine=step>0 and all(math.isclose(v,z[0]+j*step,abs_tol=1e-12) for j,v in enumerate(z)); assert affine
    result['skpr']=dict(record([[float(r['h']),float(r['x']),float(r['h'])*float(r['x'])] for r in raw]),returned_Z=z,Z_equal=z==[0,1,2,3,4],Z_affine=dict(offset=z[0],scale=step))
else: result['skpr']=dict(verdict='NOT_RUN',error=path.with_name('sessionInfo.txt').read_text() if path.with_name('sessionInfo.txt').exists() else 'Colab pending')
path.write_text(json.dumps(result,indent=2)+'\n'); print('2 PASS, 0 FAIL',result)
