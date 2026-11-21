"""G1 bounded same-host calls; reuse the crossover pool and D/PyDOE calls."""
import json, os, pathlib, subprocess, sys, time
if '--cell' in sys.argv:
    import importlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench_crossover as b
    t=b.torch; t.set_num_threads(os.cpu_count()); kind, dev, size, crit=sys.argv[2:]
    if kind=='glm':
        a=json.loads(pathlib.Path('reports/w15-glmpower-rd1b/skpr.json').read_text())['attributes']
        from torchoed import glm_power
        if dev=='cuda': t.cuda.synchronize(); t.cuda.reset_peak_memory_stats()
        start=time.perf_counter(); X=t.tensor(a['model_matrix'],dtype=t.float64,device=dev); beta=t.tensor(a['anticoef'],dtype=t.float64,device=dev)
        r=glm_power(X,beta,'binomial',alpha=a['alpha'],nsim=int(size),generator=t.Generator(device=X.device).manual_seed(0))
        if dev=='cuda': t.cuda.synchronize()
        row=dict(power=r['power'].tolist(),nonconverged=r['nonconverged'],separated=r['separated'],seconds=time.perf_counter()-start)
    else:
        k=int(kind); m=int(size)
        if crit=='D': row=b.call(k,m,dev)
        else:
            C=2*t.rand(m,k,dtype=t.float64,generator=t.Generator().manual_seed(0))-1; F=b.b.polynomial_features(C,2,max_elements=b.MAX_ELEMENTS)
            design=importlib.import_module('torchoed.d_optimal_design'); original=design._linear_exchange
            if dev=='cuda':
                def exchange(F,ids,B,*args,**kw):
                    r=original(F.cuda(),ids.cuda(),B.cuda(),*args,**kw); r['ids']=r['ids'].cpu(); return r
                design._linear_exchange=exchange; t.cuda.synchronize(); t.cuda.reset_peak_memory_stats()
            start=time.perf_counter(); r=b.b.optimal_design(F,2*F.shape[1],criterion=crit,restarts=1,generator=t.Generator().manual_seed(0),max_elements=b.MAX_ELEMENTS)
            if dev=='cuda': t.cuda.synchronize()
            row=dict(ids=r['ids'].tolist(),objective=r['objective'],returned_floor=r['efficiency_lower_bound'],seconds=time.perf_counter()-start)
    row.update(criterion=crit,threads=t.get_num_threads(),peak_gpu_bytes=t.cuda.max_memory_allocated() if dev=='cuda' else 0)
    print(json.dumps(row)); sys.exit(0)
root=pathlib.Path('/content/torchoed-g1'); sha=os.environ['SAE_CANDIDATE_SHA']
subprocess.run(['git','clone','https://github.com/JohnnyZPiggy/TorchOED.git',str(root)],check=True,capture_output=True)
subprocess.run(['git','checkout',sha],cwd=root,check=True,capture_output=True)
subprocess.run([sys.executable,'-m','pip','install','pydoe==1.6.0'],check=True,capture_output=True)
data=dict(source_sha=sha,status='PASS',rows=[],c3_glm_power=[],device_observed=json.loads(os.environ['SAE_DEVICE_OBSERVED']))
start=time.monotonic()
def cell(kind,dev,size,crit,warmup=False):
    if time.monotonic()-start>8500: return
    try:
        r=subprocess.run([sys.executable,'-B','tools/g1_driver.py','--cell',str(kind),dev,str(size),crit],cwd=root,capture_output=True,text=True,timeout=min(1200,8500-(time.monotonic()-start)))
        row=json.loads(r.stdout) if r.returncode==0 else dict(status='ERROR',error=r.stderr)
    except subprocess.TimeoutExpired: row=dict(status='TIMEOUT')
    row.update(k=kind,m=size,method=dev,criterion=crit,warmup=warmup)
    data['c3_glm_power' if kind=='glm' else 'rows'].append(row)
    pathlib.Path('/content/g1-crossover.json').write_text(json.dumps(data,allow_nan=False)+'\n')
cell('glm','cuda',10000,'binomial',True)
for nsim in (10000,100000):
    for dev in ('cpu','cuda'): cell('glm',dev,nsim,'binomial')
for k in (3,5,7): cell(k,'cuda',1000,'D',True)
for m in (1000,10000,100000,1000000):
    for k in (3,5,7):
        for crit in (('D','A','I') if k==5 else ('D',)):
            for dev in ('cpu','cuda'): cell(k,dev,m,crit)
        if m<=10000 or (m==100000 and k==3):
            for dev in ('fedorov','modified_fedorov','detmax'): cell(k,dev,m,'D')
