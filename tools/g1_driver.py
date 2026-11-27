"""G4-GLM: frozen same-host CPU/CUDA cells, resume checkpoints, bounded runtime."""
import json, os, pathlib, subprocess, sys, time
if '--cell' in sys.argv:
    sys.path.insert(0,str(pathlib.Path('src').resolve()))
    import torch as t
    from torchoed import glm_power
    t.set_num_threads(os.cpu_count())
    kind,dev,size,seed=sys.argv[2:]; N=int(size)
    a=json.loads(pathlib.Path('reports/w15-glmpower-rd1b/skpr.json').read_text())['attributes']
    if dev=='cuda': t.cuda.synchronize(); t.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    X=t.tensor(a['model_matrix'],dtype=t.float64,device=dev); beta=t.tensor(a['anticoef'],dtype=t.float64,device=dev)
    r=glm_power(X,beta,'binomial',alpha=.2,nsim=N,generator=t.Generator(device=dev).manual_seed(int(seed)),max_elements=N*(90+9))
    if dev=='cuda': t.cuda.synchronize()
    row=dict(power=r['power'].tolist(),mc_se=r['mc_se'].tolist(),nonconverged=r['nonconverged'],separated=r['separated'],seconds=time.perf_counter()-start,threads=t.get_num_threads(),peak_gpu_bytes=t.cuda.max_memory_allocated() if dev=='cuda' else 0)
    print(json.dumps(row)); sys.exit(0)
checkpoint=pathlib.Path('/content/g1-crossover.json')
if not checkpoint.exists(): checkpoint.write_text('{"status":"STARTED"}')
log=open('/content/g1-driver.log','a',buffering=1); os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
root=pathlib.Path('/content/torchoed-g1'); driver=pathlib.Path(__file__).read_bytes()
sha='94361f75319744ccaf2ebc716d08feeb2e53b9b7'
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
(root/'tools/g1_driver.py').write_bytes(driver)
import torch
saved=json.loads(checkpoint.read_text())
data=dict(status='PASS',source_sha=sha,driver_sha=os.environ['SAE_CANDIDATE_SHA'],device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version),rows=saved.get('rows',[]))
start=time.monotonic()
cells=[(10000,0,'cuda',True)]+[(N,seed,dev,False) for N,seeds in [(100000,5),(1000000,10)] for seed in range(seeds) for dev in ('cuda','cpu')]
completed={(r['nsim'],r['seed'],r['method'],r['warmup']) for r in data['rows'] if r['status']!='NOT_RUN'}
data['rows']=[r for r in data['rows'] if r['status']!='NOT_RUN']
for N,seed,dev,warmup in cells:
    if (N,seed,dev,warmup) in completed: continue
    remaining=28800-(time.monotonic()-start)
    row=dict(status='NOT_RUN',reason='internal wall budget exhausted')
    if remaining>0:
        try:
            r=subprocess.run([sys.executable,'-B','tools/g1_driver.py','--cell','glm',dev,str(N),str(seed)],cwd=root,capture_output=True,text=True,timeout=min(7200,remaining))
            row=json.loads(r.stdout) if r.returncode==0 else dict(status='ERROR',error=r.stderr)
        except subprocess.TimeoutExpired: row=dict(status='TIMEOUT')
    row.setdefault('status','PASS')
    row.update(nsim=N,seed=seed,method=dev,warmup=warmup)
    data['rows'].append(row)
    checkpoint.write_text(json.dumps(data,allow_nan=False)+'\n')
