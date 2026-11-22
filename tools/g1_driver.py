"""G1 night 2: frozen calls, bounded session, per-cell artifact writes."""
import json, os, pathlib, subprocess, sys, time
if '--cell' in sys.argv:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench_crossover as b
    t=b.torch; t.set_num_threads(os.cpu_count()); kind, dev, size=sys.argv[2:]
    if kind=='glm':
        a=json.loads(pathlib.Path('reports/w15-glmpower-rd1b/skpr.json').read_text())['attributes']
        from torchoed import glm_power
        t.cuda.synchronize(); t.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        X=t.tensor(a['model_matrix'],dtype=t.float64,device=dev); beta=t.tensor(a['anticoef'],dtype=t.float64,device=dev)
        r=glm_power(X,beta,'binomial',alpha=a['alpha'],nsim=int(size),generator=t.Generator(device=dev).manual_seed(0))
        t.cuda.synchronize()
        row=dict(power=r['power'].tolist(),nonconverged=r['nonconverged'],separated=r['separated'],seconds=time.perf_counter()-start)
    else:
        if dev=='cuda': t.cuda.reset_peak_memory_stats()
        row=b.call(int(kind),int(size),dev)
    row.update(threads=t.get_num_threads(),peak_gpu_bytes=t.cuda.max_memory_allocated() if dev=='cuda' else 0)
    print(json.dumps(row)); sys.exit(0)
root=pathlib.Path('/content/torchoed-g1'); driver=pathlib.Path(__file__).read_bytes()
sha='d869c1a394e03219eca40256c9c78f1f552bdc96'
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
(root/'tools/g1_driver.py').write_bytes(driver)
subprocess.run([sys.executable,'-m','pip','install','pydoe==1.6.0','scipy==1.18.1'],check=True,capture_output=True)
import scipy, scipy.linalg, inspect, importlib.metadata
assert 'assume_a' in inspect.signature(scipy.linalg.inv).parameters
data=dict(source_sha=sha,driver_sha=os.environ['SAE_CANDIDATE_SHA'],status='PASS',rows=[],c3_glm_power=[],device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']),environment=dict(scipy=scipy.__version__,pydoe=importlib.metadata.version('pydoe'),inv_signature=str(inspect.signature(scipy.linalg.inv))))
start=time.monotonic()
def cell(kind,dev,size,warmup=False):
    remaining=3600-(time.monotonic()-start)
    row=dict(status='NOT_RUN',reason='internal wall budget exhausted')
    if remaining>0:
        try:
            r=subprocess.run([sys.executable,'-B','tools/g1_driver.py','--cell',str(kind),dev,str(size)],cwd=root,capture_output=True,text=True,timeout=min(1200,remaining))
            row=json.loads(r.stdout) if r.returncode==0 else dict(status='ERROR',error=r.stderr)
        except subprocess.TimeoutExpired: row=dict(status='TIMEOUT')
    row.setdefault('status','PASS')
    row.update(k=kind,m=size,method=dev,criterion='binomial' if kind=='glm' else 'D',warmup=warmup)
    data['c3_glm_power' if kind=='glm' else 'rows'].append(row)
    pathlib.Path('/content/g1-crossover.json').write_text(json.dumps(data,allow_nan=False)+'\n')
cell('glm','cuda',10000,True)
cell('glm','cuda',10000)
for m in (1000,10000):
    for k in (3,5,7):
        for dev in ('fedorov','modified_fedorov','detmax'): cell(k,dev,m)
for k in (5,7):
    for dev in ('cpu','cuda'): cell(k,dev,100000)
for dev in ('cuda','cpu'): cell(3,dev,1000000)
