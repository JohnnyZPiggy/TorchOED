"""Experiment-only sampling, scaling and union break-even policies."""
def next_batch(batch, peak, total):
    if not 0 < peak <= total or batch < 1:
        raise ValueError('invalid measured peak/batch')
    return min(2*batch,max(batch+1,int(batch*.70*total/peak)))


def sample_positions(batch, limit=10000):
    count=min(batch,limit)
    if count==1:return [0]
    return [(i*(batch-1))//(count-1) for i in range(count)]


def break_even_union(rows):
    groups={}
    for row in rows:
        if row.get('status','PASS')!='PASS':continue
        key=(row['device'],row['p'],row['dtype'])
        groups.setdefault(key,{}).setdefault(row['B'],[]).append(row)
    result=[]
    from statistics import median
    for (device,p,dtype),batches in sorted(groups.items()):
        points=[dict(B=b,cpu=median(r['cpu_seconds'] for r in values),
                     gpu_with_transfer=median(r['gpu_with_transfer_seconds'] for r in values),
                     observations=len(values)) for b,values in sorted(batches.items())]
        index=next((i for i,r in enumerate(points) if r['gpu_with_transfer']<r['cpu']),None)
        interval=None if index is None else [0 if index==0 else points[index-1]['B'],points[index]['B']]
        result.append(dict(device=device,p=p,dtype=dtype,interval_open_closed=interval,
            B_prev=None if interval is None else interval[0],B=None if interval is None else interval[1],
            status='NOT_OBSERVED' if interval is None else 'OBSERVED',points=points,
            rule='Union by device/p/dtype/B; median across repeated measurements; compare including transfer'))
    return result
