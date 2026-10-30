"""G3-C exact-eigenpair CPU/CUDA observations; immutable production criterion."""
from fractions import Fraction as F
import math
import torch
from torchoed import c_optimality
from m2b_multigpu_exact import cayley, exact_case, strict_rational_geometric


def fixture_grid():
    for p in (3,10,30):
        rotations={}
        for exponent in range(15):
            try:
                spectrum=strict_rational_geometric(p,exponent)
            except ValueError:
                yield dict(status='CONSTRUCTION_BLOCKED',p=p,exponent=exponent,matrices=8,
                    reason='STRICT_RATIONAL_GEOMETRIC_RATIO_IRRATIONAL',
                    seeds=[190000+p*1000+i for i in range(8)])
                continue
            for sample in range(8):
                seed=190000+p*1000+sample
                if seed not in rotations:rotations[seed]=cayley(p,seed)
                matrix,contrasts=exact_case(rotations[seed],spectrum,seed+17)
                yield dict(status='CONSTRUCTED',p=p,exponent=exponent,seed=seed,
                           matrix=matrix,contrasts=contrasts)


def run_condition(devices=('cpu',),check=None,checkpoint=None):
    report=dict(status='IN_PROGRESS',construction_policy='STRICT_RATIONAL_GEOMETRIC_NO_APPROXIMATION',
                blocked=[],rows=[],constructed_matrices=0,blocked_matrices=0,failures=[])
    for case in fixture_grid():
        if check:check()
        if case['status']=='CONSTRUCTION_BLOCKED':
            report['blocked'].append(case);report['blocked_matrices']+=8
            continue
        report['constructed_matrices']+=1
        p,e,seed=case['p'],case['exponent'],case['seed']
        for device in devices:
            for dtype,bits in ((torch.float64,52),(torch.float32,23)):
                eps=F(1,2**bits)
                below_A=F(10**e)<1/(64*p*eps)
                matrix=torch.tensor([[float(x) for x in row] for row in case['matrix']],dtype=dtype,device=device)
                for contrast in case['contrasts']:
                    c=torch.tensor([float(x) for x in contrast['c']],dtype=dtype,device=device)
                    record=dict(p=p,exponent=e,seed=seed,device=device,dtype=str(dtype).removeprefix('torch.'),
                        contrast=contrast['name'],kappa=10**e,kappa_c=str(contrast['kappa_c']),
                        exact_q=str(contrast['quadratic']),below_A_bound=below_A)
                    try:
                        actual=float(c_optimality(matrix,c).item())
                        error=abs(F.from_float(actual)+contrast['quadratic'])/contrast['quadratic']
                        tolerance=64*p*eps*contrast['kappa_c']
                        record.update(accepted=True,score=actual,relative_error=float(error),
                            B_bound=float(tolerance),normalized_error=float(error/(eps*contrast['kappa_c'])),
                            status='PASS' if error<=tolerance else 'FAIL_B')
                    except ValueError as exc:
                        record.update(accepted=False,reason=str(exc),status='FAIL_A' if below_A else 'OBSERVED_REJECTION')
                    report['rows'].append(record)
                    if record['status'].startswith('FAIL'):
                        report['failures'].append(record)
                        report['status']='FAIL'
                        if checkpoint:checkpoint(report)
                        return report
        if checkpoint:checkpoint(report)
    report['statistics']=[]
    for device in devices:
        for dtype in ('float64','float32'):
            for contrast in ('v_max','v_min','random'):
                rows=[r for r in report['rows'] if r['device']==device and r['dtype']==dtype and r['contrast']==contrast]
                errors=sorted(r['normalized_error'] for r in rows if r['accepted'])
                report['statistics'].append(dict(device=device,dtype=dtype,contrast=contrast,
                    accepted=len(errors),rejected=sum(not r['accepted'] for r in rows),
                    max_normalized=max(errors,default=None),
                    p99_normalized=errors[max(0,math.ceil(.99*len(errors))-1)] if errors else None))
    report['disagreements']=[]
    groups={}
    for r in report['rows']:
        key=(r['p'],r['exponent'],r['seed'],r['contrast'])
        groups.setdefault(key,[]).append(r)
    for (p,e,seed,contrast),rows in groups.items():
        if len({r['accepted'] for r in rows})>1:
            report['disagreements'].append(dict(p=p,exponent=e,seed=seed,contrast=contrast,
                decisions=[{k:r[k] for k in ('device','dtype','accepted','below_A_bound')} for r in rows]))
    report['status']='PASS_CONSTRUCTIBLE_GRID_WITH_CONSTRUCTION_BLOCKS'
    return report
