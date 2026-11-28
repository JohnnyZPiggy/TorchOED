"""G4-GLM: frozen same-host CPU/CUDA cells, resume checkpoints, bounded runtime."""
import json, os, pathlib, resource, subprocess, sys, time
if '--cell' in sys.argv:
    sys.path.insert(0,str(pathlib.Path('src').resolve()))
    import torch as t
    from torchoed import glm_power, residual_bootstrap
    t.set_num_threads(os.cpu_count())
    kind,dev,size,seed=sys.argv[2:6]; N=int(size)
    dataset=sys.argv[6] if kind=='boot' else None
    a=(json.loads(pathlib.Path('reports/w19-bootstrap-refs/r.json').read_text())[dataset] if kind=='boot' else json.loads(pathlib.Path('reports/w15-glmpower-rd1b/skpr.json').read_text())['attributes'])
    if dev=='cuda': t.cuda.synchronize(); t.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    X=t.tensor(a['model_matrix'],dtype=t.float64,device=dev)
    if kind=='boot':
        y=t.tensor(a['y'],dtype=t.float64,device=dev)
        kw=dict(indices=t.tensor(a['indices'],dtype=t.int64,device=dev)-1) if seed=='oracle' else dict(generator=t.Generator(device=dev).manual_seed(int(seed)))
        r=residual_bootstrap(X,y,B=N,level=.95,max_elements=X.shape[0]*N,**kw)
    else:
        beta=t.tensor(a['anticoef'],dtype=t.float64,device=dev)
        r=glm_power(X,beta,'binomial',alpha=.2,nsim=N,generator=t.Generator(device=dev).manual_seed(int(seed)),max_elements=N*(90+9))
    if dev=='cuda': t.cuda.synchronize()
    seconds=time.perf_counter()-start
    row=(dict(lower=r['lower'].tolist(),upper=r['upper'].tolist()) if kind=='boot' else dict(power=r['power'].tolist(),mc_se=r['mc_se'].tolist(),nonconverged=r['nonconverged'],separated=r['separated']))
    row.update(seconds=seconds,threads=t.get_num_threads(),peak_gpu_bytes=t.cuda.max_memory_allocated() if dev=='cuda' else 0)
    row["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    print(json.dumps(row)); sys.exit(0)
checkpoint=pathlib.Path('/content/g1-crossover.json')
if not checkpoint.exists(): checkpoint.write_text('{"status":"STARTED"}')
log=open('/content/g1-driver.log','a',buffering=1); os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
root=pathlib.Path('/content/torchoed-g1'); driver=pathlib.Path(__file__).read_bytes()
sha='4b97ea1736707c1c095d398f698504026e542e92'
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
(root/'tools/g1_driver.py').write_bytes(driver)
import torch
saved=json.loads(checkpoint.read_text())
data=dict(status='PASS',source_sha=sha,driver_sha=os.environ['SAE_CANDIDATE_SHA'],device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version),rows=saved.get('rows',[]))
data["host_ram_bytes"]=os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")
start=time.monotonic()
attempt_boot_w19=[('stackloss',10000,0,'cuda',True)]+[(ds,999,'oracle',dev,False) for ds in ('stackloss','ChemReact') for dev in ('cuda','cpu')]+[(ds,B,seed,dev,False) for ds in ('stackloss','ChemReact') for B in (10000,100000,1000000) for seed in range(5) for dev in ('cuda','cpu')]
cells=attempt_boot_w19
completed={(r['dataset'],r['nsim'],r['seed'],r['method'],r['warmup']) for r in data['rows'] if r['status']!='NOT_RUN'}
data['rows']=[r for r in data['rows'] if r['status']!='NOT_RUN']
for dataset,N,seed,dev,warmup in cells:
    if (dataset,N,seed,dev,warmup) in completed: continue
    remaining=28800-(time.monotonic()-start)
    row=dict(status='NOT_RUN',reason='internal wall budget exhausted')
    if remaining>0:
        try:
            r=subprocess.run([sys.executable,'-B','tools/g1_driver.py','--cell','boot',dev,str(N),str(seed),dataset],cwd=root,capture_output=True,text=True,timeout=min(7200,remaining))
            row=json.loads(r.stdout) if r.returncode==0 else dict(status='ERROR',returncode=r.returncode,error=r.stderr[-2000:])
        except subprocess.TimeoutExpired: row=dict(status='TIMEOUT')
    row.setdefault('status','PASS')
    row.update(dataset=dataset,nsim=N,seed=seed,method=dev,warmup=warmup)
    data['rows'].append(row)
    checkpoint.write_text(json.dumps(data,allow_nan=False)+'\n')
