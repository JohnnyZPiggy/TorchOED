"""W12 fixed structured cases; local: python tools/w12_gpu_driver.py OUTPUT.json."""
import inspect, json, os, pathlib, subprocess, sys, time, traceback
remote = 'SAE_CANDIDATE_SHA' in os.environ
root = pathlib.Path('/content/torchoed-w362') if remote else pathlib.Path(__file__).resolve().parents[1]
sha = os.environ['SAE_CANDIDATE_SHA'] if remote else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if remote:
    subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
    subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
sys.path.insert(0,str(root/'src'))
import torch
from torchoed import optimal_design, polynomial_features
torch.set_num_threads(1)
restarts = inspect.signature(optimal_design).parameters['restarts'].default
data = dict(source_sha=sha,status='PASS',environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if remote else None,threads=1),restarts=restarts,rows=[],warmup=[])
for device in (('cpu','cuda') if remote else ('cpu',)):
    C = torch.cartesian_prod(*[torch.tensor([-1.,0.,1.],dtype=torch.float64,device=device)]*4)
    F = polynomial_features(C,degree=2)
    # hard convention: tests/test_blocked_design.py::test_split_plot_matches_frozen_exhaustive_optimum
    cases = [('B4',F[:,1:],dict(blocks=[8]*3)),('S4',F,dict(whole_plots=[4]*6,hard=C[:,0],eta=1.0))]
    def call(features,options,criterion):
        return optimal_design(features,24,criterion=criterion,generator=torch.Generator(device).manual_seed(0),**options)
    try:
        call(cases[0][1],cases[0][2],'D')
        if device=='cuda': torch.cuda.synchronize()
        data['warmup'].append(dict(device=device,status='PASS'))
    except Exception:
        data['warmup'].append(dict(device=device,status='BLOCKED',traceback=traceback.format_exc()))
    for case,features,options in cases:
        for criterion in ('D','A','I'):
            row = dict(case=case,criterion=criterion,device=device)
            try:
                if device=='cuda': torch.cuda.synchronize()
                begin=time.perf_counter(); result=call(features,options,criterion)
                if device=='cuda': torch.cuda.synchronize()
                row.update(status='PASS',seconds_per_restart=(time.perf_counter()-begin)/restarts,objective=result['objective'],floor=result['efficiency_lower_bound'],ids=result['ids'].cpu().tolist())
            except Exception:
                row.update(status='BLOCKED',traceback=traceback.format_exc())
            data['rows'].append(row)
for row in data['rows']:
    cpu = next(r for r in data['rows'] if r['device']=='cpu' and (r['case'],r['criterion'])==(row['case'],row['criterion']))
    row['ids_equal'] = sorted(row['ids'])==sorted(cpu['ids']) if row['status']==cpu['status']=='PASS' else None
    row['quality_equal'] = (row['ids_equal'] or abs(row['objective']-cpu['objective'])<=1e-9*abs(cpu['objective'])) if row['ids_equal'] is not None else None
path = pathlib.Path('/content/w12-structgpu.json' if remote else sys.argv[1])
path.write_text(json.dumps(data,allow_nan=False,indent=2)+'\n',encoding='utf-8')
print([(r['case'],r['criterion'],r['device'],r['status'],r.get('objective')) for r in data['rows']],flush=True)
