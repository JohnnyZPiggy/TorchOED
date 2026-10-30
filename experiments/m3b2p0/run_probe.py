"""Bounded PROBE_ONLY scientific runner for Issue13, not a benchmark/API release."""
import argparse
import bisect
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import random
import runpy
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from gather_scores import _Scorer, gather_two_swap_ratios
from torchoed import score_two_swaps

ORACLE = runpy.run_path(str(ROOT/'tests/oracle_two_swap.py'))
OLD_ORACLE = runpy.run_path(str(ROOT/'tests/oracle_exchange.py'))
FORMS = ('gather4','add_first')
EPS64 = torch.finfo(torch.float64).eps


def _features(rows):
    return torch.tensor([[float(Fraction(str(x))) for x in r] for r in rows],dtype=torch.float64)


def _stats(actual, reference, p, condition, enforce=True):
    expected = reference.to(device=actual.device,dtype=torch.float64)
    errors = (actual.double()-expected).abs()
    tolerance = 32*p*EPS64*max(1.,condition)*expected.abs().clamp_min(1)
    scaled = errors/tolerance
    bad = (~torch.isfinite(scaled)) | (scaled > 1)
    finite = scaled[torch.isfinite(scaled)]
    fails = bad.sum().item()
    return dict(status=('PASS' if not fails else 'FAIL') if enforce else 'OBSERVED',
                compared=actual.numel(),exceedances=fails,
                max_abs_error=errors.max().item(),max_observed_over_tol=scaled.max().item(),
                p99_observed_over_tol=torch.quantile(finite,.99).item() if finite.numel() else None,
                tolerance_formula='32*p*eps64*max(1,cond2(M))*max(1,abs(reference))',
                condition=condition,acceptance=enforce)


def _sync(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def _small(rows, selected, device, dtype, expected, form, label):
    cpu = _features(rows)
    _sync(device)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    F=cpu.to(device=device,dtype=dtype);ids=torch.tensor(selected,device=device)
    state=_Scorer(F,ids,form=form)
    tiles=list(state.tiles())
    actual=torch.cat([t[3] for t in tiles])
    positions=torch.cat([t[1] for t in tiles]).cpu()
    incoming=torch.cat([t[2] for t in tiles]).cpu()
    result=_stats(actual,torch.tensor(expected,dtype=torch.float64),F.shape[1],state.condition)
    pool=[j for j in range(len(rows)) if j not in selected]
    reference=[(po,ne) for po in combinations(range(len(selected)),2) for ne in combinations(pool,2)]
    order_ok=(positions.tolist()==[list(r[0]) for r in reference]
              and incoming.tolist()==[list(r[1]) for r in reference])
    if not order_ok:
        result['status']='FAIL'
    _sync(device)
    result.update(fixture=label,form=form,pairs=state.count,ordering_exact=order_ok,
                  wall_seconds_with_input_transfer_and_result_readback=time.perf_counter()-start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                  effective_tile_size=state.width)
    return result


def p1(device,dtype):
    fixtures=[('trap',ORACLE['TRAP'],[0,1]),('integer11',ORACLE['INTEGER11'],list(range(6))),
              ('saturated3',[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],[0,1,2]),
              ('duplicate',[[1,0],[0,1],[1,0],[0,1],[1,0]],[0,1]),
              ('singular',[[1,0],[0,1],[1,0],[2,0]],[0,1])]
    results=[]
    for label,rows,selected in fixtures:
        old=ORACLE['bareiss_det'](ORACLE['exact_gram'](rows,selected))
        ratios=[float(r[4]/old) for r in ORACLE['pair_cases'](rows,selected)]
        for form in FORMS:
            results.append(_small(rows,selected,device,dtype,ratios,form,label))
    return results


def p2(device,dtype):
    results=[]
    for p in (4,6,8):
        m,n=2*p+3,p+2
        rows=[[ (Fraction(2*i,m-1)-1)**j for j in range(p)] for i in range(m)]
        selected=[i*(m-1)//(n-1) for i in range(n)]
        old=ORACLE['bareiss_det'](ORACLE['exact_gram'](rows,selected))
        expected=[float(r[4]/old) for r in ORACLE['pair_cases'](rows,selected)]
        for form in FORMS:
            results.append(_small(rows,selected,device,dtype,expected,form,'monomial_p'+str(p)))
    return results


def p3(device,dtype):
    g=torch.Generator().manual_seed(13003)
    cpu=torch.randint(-2,3,(60,6),generator=g,dtype=torch.int64).double()
    selected=torch.arange(12)
    start=time.perf_counter()
    reference=score_two_swaps(cpu,selected)
    reference_seconds=time.perf_counter()-start
    assert reference['evaluated_pairs']==74448
    mask=reference['status']==0
    results=[]
    for form in FORMS:
        _sync(device)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        F,ids=cpu.to(device=device,dtype=dtype),selected.to(device)
        state=_Scorer(F,ids,form=form)
        tiles=list(state.tiles())
        actual=torch.cat([t[3] for t in tiles])
        positions=torch.cat([t[1] for t in tiles]).cpu()
        incoming=torch.cat([t[2] for t in tiles]).cpu()
        order_ok=(torch.equal(positions,reference['outgoing_positions'])
                  and torch.equal(incoming,reference['incoming_ids']))
        result=_stats(actual[mask.to(device)],reference['ratios'][mask],6,state.condition)
        if not order_ok:
            result['status']='FAIL'
        _sync(device)
        result.update(form=form,pairs=state.count,ordering_exact=order_ok,
                      reference_status_counts=reference['counts'],reference_cpu_seconds=reference_seconds,
                      wall_seconds_with_input_transfer_and_result_readback=time.perf_counter()-start,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        results.append(result)
    return results


def _fraction_solve(M,B):
    n=len(M);a=[list(M[i])+list(B[i]) for i in range(n)]
    for k in range(n):
        pivot=next(i for i in range(k,n) if a[i][k])
        a[k],a[pivot]=a[pivot],a[k]
        value=a[k][k];a[k]=[x/value for x in a[k]]
        for i in range(n):
            if i!=k:
                factor=a[i][k];a[i]=[x-factor*y for x,y in zip(a[i],a[k])]
    return [r[n:] for r in a]


def p4(device,dtype):
    rows=ORACLE['INTEGER11'];selected=list(range(6));p=3
    F=[[Fraction(x) for x in row] for row in rows]
    M=ORACLE['exact_gram'](rows,selected)
    solved=_fraction_solve(M,list(map(list,zip(*F))))
    G=[[sum(F[i][k]*solved[k][j] for k in range(p)) for j in range(len(F))] for i in range(len(F))]
    denominator=ORACLE['bareiss_det'](M);count=0
    for i,j,numerator in OLD_ORACLE['neighborhood'](rows,selected):
        old=selected[i]
        assert (1+G[j][j])*(1-G[old][old])+G[old][j]**2==numerator/denominator
        count+=1
    return [dict(status='PASS',exact_swaps=count,method='independent Fraction solve/delta/direct Gram',production_k1_search='NOT_RUN')]


def p5(device,dtype):
    F=_features(ORACLE['INTEGER11']);ids=torch.arange(6)
    invalid=[(F.float(),ids,{}),(F,ids.float(),{}),(F,torch.tensor([0,0,1,2,3,4]),{}),
             (F,torch.tensor([-1,0,1,2,3,4]),{}),(F,torch.tensor([0,1,2,3,4,11]),{}),
             (torch.ones(11,3,dtype=torch.float64),ids,{}),(F,ids,{'max_elements':10}),
             (F,ids,{'form':'delete_first'}),(F,ids,{'tile_size':0})]
    for features,selected,options in invalid:
        try:
            gather_two_swap_ratios(features,selected,**options)
        except (ValueError,TypeError):
            continue
        raise AssertionError('negative input accepted')
    full=torch.cat(list(gather_two_swap_ratios(F,ids,tile_size=7)))
    tiled_state=_Scorer(F,ids,tile_size=7,max_elements=990)
    assert tiled_state.nn is None
    fallback=torch.cat([x[3] for x in tiled_state.tiles()])
    tolerance=32*3*EPS64*max(1,tiled_state.condition)*full.abs().clamp_min(1)
    assert bool(((fallback-full).abs()<=tolerance).all())
    return [dict(status='PASS',rejections=len(invalid),bounded_G_NN_cache='PASS',workspace_elements=tiled_state.workspace_elements)]


def _sample_reference(cpu,selected,ranks):
    pool=[i for i in range(cpu.shape[0]) if i not in selected.tolist()]
    outgoing=list(combinations(range(selected.numel()),2));q=len(pool);c=q*(q-1)//2
    prefix=[i*(2*q-i-1)//2 for i in range(q)]
    pos=[];new=[]
    for rank in ranks:
        i,j=outgoing[rank//c];r=rank%c
        a=bisect.bisect_right(prefix,r)-1;b=a+1+r-prefix[a]
        pos.append((i,j));new.append((pool[a],pool[b]))
    pos,new=torch.tensor(pos),torch.tensor(new)
    denominator=torch.linalg.det(cpu[selected].mT@cpu[selected])
    expected=[]
    for start in range(0,len(ranks),1000):
        positions,incoming=pos[start:start+1000],new[start:start+1000]
        trial=selected.expand(positions.shape[0],-1).clone()
        trial.scatter_(1,positions,incoming)
        X=cpu[trial]
        expected.append(torch.linalg.det(X.mT@X)/denominator)
    return pos,new,torch.cat(expected)


def _stream_compare(a,b,enforce):
    ia,ib=iter(a.tiles()),iter(b.tiles());A,B=next(ia),next(ib);oa=ob=0;count=0
    device=a.F.device
    worst=torch.zeros((),device=device,dtype=torch.float64)
    maxscaled=worst.clone();failures=torch.zeros((),device=device,dtype=torch.int64);badorder=failures.clone()
    while A is not None and B is not None:
        length=min(A[3].numel()-oa,B[3].numel()-ob)
        xa,xb=A[3][oa:oa+length].double(),B[3][ob:ob+length].double()
        err=(xa-xb).abs();tol=32*a.p*EPS64*max(1,a.condition)*xb.abs().clamp_min(1)
        scaled=err/tol
        worst=torch.maximum(worst,err.max());maxscaled=torch.maximum(maxscaled,scaled.max())
        failures+=((~torch.isfinite(scaled))|(scaled>1)).sum()
        badorder+=(A[1][oa:oa+length]!=B[1][ob:ob+length]).sum()+(A[2][oa:oa+length]!=B[2][ob:ob+length]).sum()
        count+=length;oa+=length;ob+=length
        if oa==A[3].numel():
            A=next(ia,None);oa=0
        if ob==B[3].numel():
            B=next(ib,None);ob=0
    assert A is None and B is None and count==a.count==b.count
    errors,order=failures.item(),badorder.item()
    return dict(status=('PASS' if errors==order==0 else 'FAIL') if enforce else 'OBSERVED',
                pairs=count,index_mismatches=order,exceedances=errors,
                max_abs_error=worst.item(),max_observed_over_tol=maxscaled.item(),
                percentile_scope='independent 10k subsample below; full stream stores maxima/counts only')


def p6(device,dtype,progress=None):
    results=[];enforce=dtype==torch.float64
    for m in (200,1000):
        generator=torch.Generator().manual_seed(13006+m)
        cpu=torch.randn(m,10,generator=generator,dtype=torch.float64);selected=torch.arange(20)
        count=math.comb(20,2)*math.comb(m-20,2)
        ranks=sorted(random.Random(130060+m).sample(range(count),10000))
        expected_pos,expected_new,expected=_sample_reference(cpu,selected,ranks)
        for form in FORMS:
            _sync(device)
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            F=cpu.to(device=device,dtype=dtype);ids=selected.to(device)
            a=_Scorer(F,ids,form=form,tile_size=8192,observed_float32=not enforce)
            b=_Scorer(F,ids,form=form,tile_size=16384,observed_float32=not enforce)
            stream=_stream_compare(a,b,enforce)
            values=[];positions=[];incoming=[]
            for offset in range(0,len(ranks),a.width):
                rank_tensor=torch.tensor(ranks[offset:offset+a.width],device=device)
                po,ne,ra=a.evaluate(rank_tensor)
                positions.append(po.cpu());incoming.append(ne.cpu());values.append(ra)
            order_ok=(torch.equal(torch.cat(positions),expected_pos) and torch.equal(torch.cat(incoming),expected_new))
            sample=_stats(torch.cat(values),expected,10,a.condition,enforce)
            if enforce and not order_ok:
                sample['status']='FAIL'
            _sync(device)
            status='OBSERVED' if not enforce else ('PASS' if stream['status']==sample['status']=='PASS' else 'FAIL')
            results.append(dict(status=status,m=m,n=20,p=10,form=form,pairs=count,
                effective_tiles=[a.width,b.width],G_NN_storage='full' if a.nn is not None else 'row_cached',
                stream_comparison=stream,independent_direct_Gram_sample=sample,sample_ordering_exact=order_ok,
                sample_seed=130060+m,sample_count=len(ranks),dtype=str(dtype),
                wall_seconds_with_input_transfer_and_result_readback=time.perf_counter()-start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
            if progress:
                progress(results)
    return results


def _safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_safe(v) for v in value]
    return value


def run(gates,device,dtype,source_sha,threads=4,output=None):
    torch.set_num_threads(threads)  # process-local experiment setting, recorded below
    device=torch.device(device);dtype=getattr(torch,dtype)
    if dtype!=torch.float64 and list(gates)!=['P6']:
        raise ValueError('float32 is P6 OBSERVED only')
    report=dict(source_sha=source_sha,status='RUNNING',claim='PROBE_ONLY',device=str(device),
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),
        threads=torch.get_num_threads(),dtype=str(dtype),gates={},
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        numerical_policy='calibrated hypothesis, not rigorous interval/SPD/optimizer certificate',
        timing_scope='input H2D + prepare + scoring + reductions/result D2H; excludes precomputed reference; PROBE_ONLY')
    def save():
        if output:
            Path(output).write_text(json.dumps(_safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    functions=dict(P1=p1,P2=p2,P3=p3,P4=p4,P5=p5,P6=p6)
    for gate in gates:
        print('START',gate,device,dtype,flush=True)
        start=time.perf_counter()
        try:
            if gate=='P6':
                def checkpoint(values):
                    report['gates'][gate]=dict(status='RUNNING',cases=values);save()
                results=p6(device,dtype,checkpoint)
            else:
                results=functions[gate](device,dtype)
            status='OBSERVED' if dtype==torch.float32 else ('PASS' if all(r['status']=='PASS' for r in results) else 'FAIL')
            report['gates'][gate]=dict(status=status,cases=results,elapsed_seconds=time.perf_counter()-start)
        except Exception as exc:
            report['gates'][gate]=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        save()
        print('END',gate,report['gates'][gate]['status'],flush=True)
    report['status']='FAIL' if any(x['status']=='FAIL' for x in report['gates'].values()) else 'PASS'
    report['float32_acceptance']='OBSERVED_ONLY' if dtype==torch.float32 else 'NOT_RUN'
    report['code_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),Path(__file__).with_name('gather_scores.py')]}
    save()
    return _safe(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gates',nargs='+',choices=['P1','P2','P3','P4','P5','P6'],default=['P1','P2','P3','P4','P5','P6'])
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dtype',default='float64',choices=['float64','float32'])
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=run(args.gates,args.device,args.dtype,args.source_sha,args.threads,args.out)
    raise SystemExit(0 if result['status']=='PASS' else 1)
