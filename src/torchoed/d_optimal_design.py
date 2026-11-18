"""Seeded feasible random starts followed by CPU D-optimal VND restarts."""
import torch

from ._validation import integer
from .criteria import d_optimality
from .exchange import _tolerance
from .information import exact_information, _features
from .k2_exchange import k2_exchange, _verified_information
from .two_swap import _pair_tiles


def _fixed_pool(fixed_ids, m, n):
    fixed = torch.as_tensor(fixed_ids)
    if (fixed.ndim != 1 or len(fixed) >= n or
            (len(fixed) and (fixed.dtype not in (torch.int32, torch.int64) or
             fixed.min() < 0 or fixed.max() >= m or len(fixed.unique()) != len(fixed)))):
        raise ValueError('fixed_ids must be distinct candidate IDs with length < n')
    fixed = fixed.to(dtype=torch.int64)
    return fixed, torch.tensor(sorted(set(range(m))-set(fixed.tolist())),dtype=torch.int64)


def _restarts(restarts, attempt):
    starts, objectives, failures, best = [], [], 0, None
    best_restart = None
    for restart in range(restarts):
        while (trial := attempt()) is None:
            failures += 1
            if failures >= 100 * restarts:
                raise ValueError("No feasible start after 100*restarts failed draws; "
                                 "candidate set may be rank-deficient for this n") from None
        ids, result, value = trial
        starts.append(ids.tolist())
        if best is None or value > objectives[best_restart]:
            best, best_restart = result, restart
        objectives.append(value)
    return best, best_restart, starts, objectives


def _dual_floor(F, M, n, criterion, current):
    m, p = F.shape
    w = F.new_full((m,),1/m)
    B = torch.eye(p,dtype=F.dtype,device=F.device) if criterion=='A' else F.mT@F/m
    R = torch.linalg.cholesky(B).mT if criterion=='I' else None
    own = torch.linalg.slogdet(M/n).logabsdet if criterion=='D' else n*torch.trace(torch.linalg.solve(M,B))
    best = current
    for iteration in range(1000):
        L, info = torch.linalg.cholesky_ex(F.mT@(w[:,None]*F))
        Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
        if criterion=='D':
            scores = Z.square().sum(0)
            ratio = p/scores.max()
            floor = ((own-2*L.diagonal().log().sum())/p).exp()*ratio
            update = scores/p
        else:
            Y = torch.linalg.solve_triangular(L.mT,Z,upper=True)
            scores = (Y if criterion=='A' else R@Y).square().sum(0)
            phi = torch.trace(torch.cholesky_solve(B,L))
            ratio = phi/scores.max()
            floor = phi*ratio/own
            update = scores
        info_v, floor_v, ratio_v = torch.stack((info.to(F.dtype),floor,ratio)).tolist()
        if info_v:
            break
        best = max(best,floor_v)
        if ratio_v>=1-(1e-4 if criterion=='D' else 1e-3):
            break
        w = w*update
        w = w/w.sum()
    return best


@torch.no_grad()
def d_optimal_design(F: torch.Tensor, n: int, *, fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None, max_iter: int = 100,
                     tile_size: int = 64, max_elements: int = 1_000_000) -> dict[str, object]:
    """Return the best restarted k2_exchange run; see docs/M3C_API.md."""
    return _exact_design(F, n, "D", fixed_ids, restarts, generator, max_iter, max_elements, tile_size)


def _linear_scores(F, ids, B, criterion):
    M = F[ids].mT@F[ids]
    L = torch.linalg.cholesky(M)
    Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
    if criterion == "A":
        Y = torch.linalg.solve_triangular(L.mT,Z,upper=True)
        Qs, diagonal = Y[:,ids].mT@Y, Y.square().sum(dim=0)
    else:
        HZ = (Z@Z.mT/len(F))@Z
        Qs, diagonal = Z[:,ids].mT@HZ, (Z*HZ).sum(dim=0)
    Gs, gd = Z[:,ids].mT@Z, Z.square().sum(dim=0)
    incoming = torch.tensor(sorted(set(range(len(F)))-set(ids.tolist())))
    u, v, c = Gs[:,ids].diagonal()[:,None], gd[incoming], Gs[:,incoming]
    alpha, beta, gamma = Qs[:,ids].diagonal()[:,None], diagonal[incoming], Qs[:,incoming]
    ratio = (1-u)*(1+v)+c.square()
    delta = ((1+v)*alpha-2*c*gamma-(1-u)*beta)/ratio
    phi = torch.trace(torch.cholesky_solve(B,L)).item()
    return M, phi, diagonal, incoming, ratio, delta


def _linear_closed_terms(G, Q, incoming=None):
    if incoming is None:
        gaa,gab,gbb,gcc,w,gdd,gac,gad,gbc,gbd = G
        qaa,qab,qbb,qcc,qcd,qdd,qac,qad,qbc,qbd = Q
        u, v = 1+gcc, 1+gdd
        pi = u*v-w*w
        addition = (v*qcc+u*qdd-2*w*qcd)/pi
    else:
        gaa,gab,gbb,gac,gad,gbc,gbd = G
        qaa,qab,qbb,qac,qad,qbc,qbd = Q
        u,v,w,pi,qcc,qcd,qdd,addition = incoming
    e00 = (gac*v-gad*w)/pi
    e01 = (gad*u-gac*w)/pi
    e10 = (gbc*v-gbd*w)/pi
    e11 = (gbd*u-gbc*w)/pi
    t00 = gaa-1-e00*gac-e01*gad
    t01 = gab-e00*gbc-e01*gbd
    t11 = gbb-1-e10*gbc-e11*gbd
    r00 = qaa-2*(e00*qac+e01*qad)+e00**2*qcc+2*e00*e01*qcd+e01**2*qdd
    r01 = (qab-e00*qbc-e01*qbd-e10*qac-e11*qad
           +e00*e10*qcc+(e00*e11+e01*e10)*qcd+e01*e11*qdd)
    r11 = qbb-2*(e10*qbc+e11*qbd)+e10**2*qcc+2*e10*e11*qcd+e11**2*qdd
    return addition, t11*r00+t00*r11-2*t01*r01, t00*t11-t01*t01, pi


def _linear_pair_entries(Z, T, c, d, criterion):
    g = (Z[:,c]*Z[:,d]).sum(0)
    q = ((T if criterion=='A' else Z)[:,c]*T[:,d]).sum(0)
    return g,q


def _linear_products(Z, T, ids, criterion, sliced):
    if sliced:
        G = Z[:,ids].mT@Z
        Q = T[:,ids].mT@T if criterion=='A' else Z[:,ids].mT@T
        return G,Q,Z.square().sum(0)
    return Z.mT@Z,T.mT@T if criterion=='A' else Z.mT@T,None


def _linear_screen(G, Q, idx, tolerance, cutoff, best, tables, flat, slices=None):
    if slices is not None:
        Z,T,gd,qd,criterion,positions = slices
        a,b,c,d = idx.unbind(1)
        ra,rb = positions.unbind(1)
        gcd,qcd = _linear_pair_entries(Z,T,c,d,criterion)
        addition,numerator,tau,pi = _linear_closed_terms(
            (G[ra,a],G[ra,b],G[rb,b],gd[c],gcd,gd[d],G[ra,c],G[ra,d],G[rb,c],G[rb,d]),
            (Q[ra,a],Q[ra,b],Q[rb,b],qd[c],qcd,qd[d],Q[ra,c],Q[ra,d],Q[rb,c],Q[rb,d]))
    elif tables is None:
        row = idx[:,(0,0,1,2,2,3,0,0,1,1)]
        col = idx[:,(0,1,1,2,3,3,2,3,2,3)]
        addition, numerator, tau, pi = _linear_closed_terms(G[row,col].unbind(1),Q[row,col].unbind(1))
    else:
        incoming, outgoing = tables
        out, local = flat//incoming.shape[1], flat%incoming.shape[1]
        gaa,gab,gbb,qaa,qab,qbb = outgoing[:,out].unbind(0)
        row, col = idx[:,(0,0,1,1)], idx[:,(2,3,2,3)]
        addition, numerator, tau, pi = _linear_closed_terms(
            (gaa,gab,gbb,*G[row,col].unbind(1)),(qaa,qab,qbb,*Q[row,col].unbind(1)),incoming[:,local].unbind(0))
    closed = addition+numerator/tau
    determinant = pi*tau
    band = (determinant.abs()<=1e-8) | ~torch.isfinite(determinant) | ~torch.isfinite(closed)
    # The state tolerance is exactly the gate's delta (exchange._tolerance).
    keep = (closed>tolerance-tolerance/2) & (closed>=best-tolerance)
    if cutoff is not None:
        keep &= closed<=-cutoff[0]+tolerance/2
    return keep | band


def _linear_gather4(Z, T, ids, pairs, criterion, J):
    idx = torch.cat((ids[pairs[:,:2]],pairs[:,2:]),dim=1)
    U, V = Z[:,idx].permute(1,2,0), T[:,idx].permute(1,2,0)
    K = U@U.mT+J
    Q = V@V.mT if criterion=='A' else U@V.mT
    legal = torch.linalg.det(K)>0
    gain = torch.linalg.solve(K[legal],Q[legal]).diagonal(dim1=-2,dim2=-1).sum(-1)
    return legal, gain


def _linear_eligible(pairs, gains, tolerance, cutoff):
    keep = gains>tolerance
    if cutoff is not None:
        row = pairs.new_tensor(cutoff[1])
        first = (pairs != row).int().argmax(dim=1)
        later = pairs.gather(1,first[:,None]).flatten() > row[first]
        keep &= (gains < -cutoff[0]) | ((gains == -cutoff[0]) & later)
    return keep


def _linear_prefix_limit(negative, outgoing, phi, tolerance, best, incoming=None):
    theta = min(max(tolerance,best)-tolerance,phi/2)
    shift = phi*theta/(phi-theta)
    if incoming is not None:
        return (incoming>=outgoing+shift-tolerance).long()
    return torch.searchsorted(negative,-outgoing-shift+tolerance,right=True)


def _linear_prefix_tiles(n, prefix, selected_scores, phi, tolerance, width, get_best):
    pool, negative, order = prefix
    q, count = len(pool), len(order)
    # At the capped threshold no pair can be pruned: preserve the original raw sequence.
    if -negative[-1]>=selected_scores.topk(2).values.sum()+phi-tolerance:
        offset = 0
        for pairs in _pair_tiles(n,pool.tolist(),width):
            yield pairs,torch.arange(offset,offset+len(pairs))
            offset += len(pairs)
        return
    ends = torch.arange(q).flip(0)[:-1].cumsum(0)
    i,k = torch.triu_indices(n,n,offset=1)
    outgoing = selected_scores[i]+selected_scores[k]
    start, previous = torch.zeros_like(i), None
    while True:
        best = get_best()
        if best!=previous:
            limits = _linear_prefix_limit(negative,outgoing,phi,tolerance,best)
            previous = best
        remaining = (limits-start).clamp_min(0)
        boundaries = remaining.cumsum(0)
        total = boundaries[-1].item()
        if not total:
            return
        flat = torch.arange(min(width,total))
        out = torch.searchsorted(boundaries,flat,right=True)
        rank = flat-(boundaries[out]-remaining[out])+start[out]
        local = order[rank].long()
        a = torch.searchsorted(ends,local,right=True)
        b = a+1+local-(ends[a]-(q-a-1))
        last = out[-1].item()
        start[:last],start[last] = limits[:last],rank[-1]+1
        yield torch.stack((i[out],k[out],pool[a],pool[b]),dim=1),out*count+local


def _linear_removal_terms(g, q, cg, cq, diagonal):
    u,v,w = 1-g[0],1-g[2],g[1]
    rho = u*v-w*w
    h0,h1 = (v*cg[0]+w*cg[1])/rho,(w*cg[0]+u*cg[1])/rho
    delta = (v*q[0]+2*w*q[1]+u*q[2])/rho
    t = diagonal+2*(h0*cq[0]+h1*cq[1])+h0*h0*q[0]+2*h0*h1*q[1]+h1*h1*q[2]
    return rho,delta,t


def _linear_removal_data(G, Q, chosen, pool, diagonal, rows=None):
    a,b = chosen
    r,t = (a,b) if rows is None else rows
    g,q = (G[r,a],G[r,b],G[t,b]),(Q[r,a],Q[r,b],Q[t,b])
    rho = (1-g[0])*(1-g[2])-g[1]*g[1]
    # Invalid removals use only (d); avoid dividing by their singular determinant.
    safe = tuple(torch.where(rho>=1e-6,v,0)[:,None] for v in g)
    _,delta,t = _linear_removal_terms(safe,tuple(v[:,None] for v in q),
                                    (G[r[:,None],pool],G[t[:,None],pool]),
                                    (Q[r[:,None],pool],Q[t[:,None],pool]),diagonal[pool])
    return rho,delta[:,0],t


def _linear_removal_limit(negative, phi_r, delta, theta, tolerance):
    eta = torch.minimum(theta+delta,phi_r/2)
    threshold = phi_r*eta/(phi_r-eta)-tolerance
    return torch.searchsorted(negative,-negative-threshold[:,None],right=True)


def _linear_square_terms(q, p):
    det = q[0]*q[2]-q[1]*q[1]
    return (q[2]*p[0]+q[0]*p[2]-2*q[1]*p[1])/det,det,q[0]+q[2]


def _linear_square_data(G, Q, chosen, pool, P, rows=None):
    a,b = chosen
    r,t = (a,b) if rows is None else rows
    alpha,det,trace = _linear_square_terms((Q[r,a],Q[r,b],Q[t,b]),P)
    scores = G[r[:,None],pool].square()+G[t[:,None],pool].square()
    return scores,alpha,trace+2*det.clamp_min(0).sqrt()


def _linear_square_limit(negative, alpha, kappa, theta, tolerance):
    den = alpha-theta+tolerance
    threshold = torch.where(den>0,kappa/den,float('inf'))
    return torch.searchsorted(negative,-negative-threshold[:,None],right=True)


def _linear_removal_tiles(prefix, selected, diagonal, G, Q, phi, tolerance, width, get_best, block_size, square=None, selected_rows=None):
    pool,negative,order = prefix
    q = len(pool)
    count = q*(q-1)//2
    i,k = torch.triu_indices(len(selected),len(selected),offset=1)
    positions = torch.arange(q)
    for offset in range(0,len(i),block_size):
        oi,ok = i[offset:offset+block_size],k[offset:offset+block_size]
        chosen = torch.stack((selected[oi],selected[ok]))
        rows = None if selected_rows is None else selected_rows[torch.stack((oi,ok))]
        outgoing = diagonal[chosen].sum(0)
        theta = min(max(tolerance,get_best())-tolerance,phi/2)
        if square is None:
            dlimit = _linear_prefix_limit(negative,outgoing,phi,tolerance,get_best())
            rho,delta,t = _linear_removal_data(G,Q,chosen,pool,diagonal,rows)
            values,permutation = t.sort(dim=1,descending=True,stable=True)
            upper = _linear_removal_limit(-values,phi+delta,delta,theta,tolerance)
            upper = torch.where((rho>=1e-6)[:,None],upper,q)
            del rho,delta
        else:
            dlimit = outgoing+phi*theta/(phi-theta)<=diagonal[pool].topk(2).values.sum()+tolerance
            t,alpha,kappa = _linear_square_data(G,Q,chosen,pool,(square[oi,oi],square[oi,ok],square[ok,ok]),rows)
            values,permutation = t.sort(dim=1,descending=True,stable=True)
            upper = _linear_square_limit(-values,alpha,kappa,theta,tolerance)
            del alpha,kappa
        lengths = (upper-positions-1).clamp_min(0)
        unpruned = lengths.sum(1)==count
        permutation = torch.where(unpruned[:,None],positions,permutation)
        lengths = torch.where(unpruned[:,None],q-positions-1,lengths)
        lengths = torch.where((dlimit>0)[:,None],lengths,0).flatten()
        boundaries = lengths.cumsum(0)
        total = boundaries[-1].item()
        del chosen,t,values,upper,unpruned
        for start in range(0,total,width):
            rank = torch.arange(start,min(start+width,total))
            run = torch.searchsorted(boundaries,rank,right=True)
            out,left = run//q,run%q
            right = left+1+rank-(boundaries[run]-lengths[run])
            a,b = permutation[out,left],permutation[out,right]
            a,b = torch.minimum(a,b),torch.maximum(a,b)
            theta = min(max(tolerance,get_best())-tolerance,phi/2)
            incoming = diagonal[pool[a]]+diagonal[pool[b]]
            keep = (_linear_prefix_limit(negative,outgoing[out],phi,tolerance,get_best(),incoming).bool()
                    if square is None else incoming>=outgoing[out]+phi*theta/(phi-theta)-tolerance)
            out,a,b = out[keep],a[keep],b[keep]
            local = a*(2*q-a-1)//2+b-a-1
            yield torch.stack((oi[out],ok[out],pool[a],pool[b]),dim=1),(offset+out)*count+local


def _linear_pair_scores(F, ids, criterion, max_elements, free_positions=None, *, tolerance=None, cutoff=None):
    m, p = F.shape
    free = torch.arange(len(ids)) if free_positions is None else free_positions.nonzero().flatten()
    incoming = sorted(set(range(m))-set(ids.tolist()))
    if len(free)<2 or len(incoming)<2:
        return
    fixed = 8*m*p+16*len(ids)*m+8*p*p
    capacity = (max_elements-fixed)//(32*p+256)
    if capacity<1:
        raise ValueError('insufficient max_elements for one linear pair tile')
    L = torch.linalg.cholesky(F[ids].mT@F[ids])
    Z = torch.linalg.solve_triangular(L,F.mT,upper=False)
    T = torch.linalg.solve_triangular(L.mT,Z,upper=True) if criterion=='A' else (Z@Z.mT/m)@Z
    J = torch.diag(F.new_tensor([-1.,-1.,1.,1.]))
    count, outgoing_count = len(incoming)*(len(incoming)-1)//2, len(free)*(len(free)-1)//2
    compact = count<=torch.iinfo(torch.int32).max
    prefix_slots = count+(count+1)//2 if compact else 2*count
    available_base = max_elements-fixed-2*len(incoming)**2-prefix_slots-8*outgoing_count
    removal, square = False, None
    table_elements = 8*count+6*outgoing_count
    # Packing indices must not activate extra tables and reduce an existing tile.
    hoisted = table_elements<=(available_base-(2*count-prefix_slots))//2
    sliced = not hoisted
    if hoisted and len(ids)>p+1:
        ebase = available_base+prefix_slots
        block = min(outgoing_count,max(1,ebase//(2*(16*len(incoming)+64))))
        sliced = table_elements>(ebase-block*(16*len(incoming)+64))//2
    row_savings = 2*len(incoming)**2-2*len(ids)*m-2*m
    if not hoisted:
        available_base += row_savings
    pair_slots = 96+4*p if not hoisted else 96
    available = available_base-table_elements if hoisted else available_base
    tables = None
    screened_capacity = available//pair_slots
    screened = tolerance is not None and screened_capacity>=1 and max_elements-fixed>=6*count+8*outgoing_count
    if screened:
        # Closed temporaries die before gather4; retain at most two sets of row IDs.
        rescore_capacity = (available-8*screened_capacity)//(32*p+256)
        screened = rescore_capacity>=1
    if screened:
        capacity = screened_capacity
        diagonal = T.square().sum(0) if criterion=='A' else (Z*T).sum(0)
        B = torch.eye(p,dtype=F.dtype) if criterion=='A' else F.mT@F/m
        phi = torch.trace(torch.cholesky_solve(B,L)).item()
        pool = torch.tensor(incoming)

        def pair_prefix():
            a,b = torch.triu_indices(len(pool),len(pool),offset=1)
            values, order = (diagonal[pool[a]]+diagonal[pool[b]]).sort(descending=True,stable=True)
            return pool,-values,order.to(torch.int32) if compact else order

        tile_slack = 96*(capacity-min(capacity,count*outgoing_count))
        block_size = min(outgoing_count,(prefix_slots+4*outgoing_count+tile_slack)//(8*len(pool)+16))
        if len(ids)==p and block_size>=1:
            # Release the d-prefix reserve; half the outgoing reserve holds block metadata.
            prefix,removal = (pool,None,None),True
        elif len(ids)<=p+1:
            prefix = pair_prefix()
        else:
            prefix = None
        G,Q,gd = _linear_products(Z,T,ids,criterion,sliced)
        if removal:
            QS = Q[free if sliced else ids[free]][:,ids]
            square = QS@QS.mT
            del QS
        if len(ids)>p+1:
            i,k = torch.triu_indices(len(free),len(free),offset=1)
            chosen = ids[free]
            r = free if sliced else chosen
            rho = (1-G[r[i],chosen[i]])*(1-G[r[k],chosen[k]])-G[r[i],chosen[k]]**2
            top = diagonal[pool].topk(2).values.sum().reshape(1)
            if ((rho>=1e-6)&(diagonal[chosen[i]]+diagonal[chosen[k]]<=top[0]+tolerance)).any():
                if sliced and hoisted:
                    available_base += row_savings
                pair_slots = 96+4*p if sliced else 96
                ebase = available_base+prefix_slots
                block_size = min(outgoing_count,max(1,ebase//(2*(16*len(pool)+64))))
                ebase -= block_size*(16*len(pool)+64)
                ehoisted = not sliced and table_elements<=ebase//2
                eavailable = ebase-table_elements if ehoisted else ebase
                ecapacity = eavailable//pair_slots
                erescore = (eavailable-8*ecapacity)//(32*p+256)
                if ecapacity>=1 and erescore>=1:
                    removal,hoisted,capacity,rescore_capacity = True,ehoisted,ecapacity,erescore
                    prefix = pool,-top,None
            del i,k,chosen,rho,top,r
        if prefix is None:
            del G,Q,gd
            prefix = pair_prefix()
            sliced = not hoisted
            G,Q,gd = _linear_products(Z,T,ids,criterion,sliced)
        if hoisted:
            c,d = torch.combinations(torch.tensor(incoming),r=2).mT
            u,v,w = 1+G[c,c],1+G[d,d],G[c,d]
            pi = u*v-w*w
            qcc,qcd,qdd = Q[c,c],Q[c,d],Q[d,d]
            incoming_table = torch.stack((u,v,w,pi,qcc,qcd,qdd,(v*qcc+u*qdd-2*w*qcd)/pi))
            a,b = torch.combinations(ids[free],r=2).mT
            outgoing_table = torch.stack((G[a,a],G[a,b],G[b,b],Q[a,a],Q[a,b],Q[b,b]))
            del a,b,c,d,u,v,w,pi,qcc,qcd,qdd
            tables = incoming_table,outgoing_table
    best = -float('inf')
    tiles = (_linear_prefix_tiles(len(free),prefix,diagonal[ids[free]],phi,tolerance,capacity,lambda:best)
             if screened else ((pairs,None) for pairs in _pair_tiles(len(free),incoming,capacity)))
    if screened and removal:
        tiles = _linear_removal_tiles(prefix,ids[free],diagonal,G,Q,phi,tolerance,capacity,lambda:best,block_size,square,free if sliced else None)
    for pairs,flat in tiles:
        pairs[:,:2] = free[pairs[:,:2]]
        if screened:
            idx = torch.cat((ids[pairs[:,:2]],pairs[:,2:]),dim=1)
            slices = (Z,T,gd,diagonal,criterion,pairs[:,:2]) if sliced else None
            pairs = pairs[_linear_screen(G,Q,idx,tolerance,cutoff,best,tables,flat,slices)]
            del idx,slices
            if not len(pairs):
                continue
        for chunk in pairs.split(rescore_capacity) if screened else (pairs,):
            legal, gain = _linear_gather4(Z,T,ids,chunk,criterion,J)
            if screened:
                eligible = _linear_eligible(chunk[legal],gain,tolerance,cutoff) & torch.isfinite(gain)
                if eligible.any():
                    best = max(best,gain[eligible].max().item())
            yield chunk[legal],gain


def _linear_pair_moves(F, ids, criterion, max_elements, free_positions, tolerance):
    cutoff = None
    while True:
        best = None
        for pairs,gains in _linear_pair_scores(F,ids,criterion,max_elements,free_positions,
                                             tolerance=tolerance,cutoff=cutoff):
            keep = _linear_eligible(pairs,gains,tolerance,cutoff)
            eligible = keep.nonzero().flatten()
            if len(eligible):
                index = eligible[gains[eligible].argmax()]
                tied = eligible[gains[eligible]==gains[index]]
                if len(tied)>1:
                    for column in (3,2,1,0):
                        tied = tied[pairs[tied,column].argsort(stable=True)]
                    index = tied[0]
                key = (-gains[index].item(),tuple(pairs[index].tolist()))
                if best is None or key<best:
                    best = key
        if best is None:
            return
        # A failed direct verification resumes the exact ordering with bounded memory.
        cutoff = best
        yield list(best[1][:2]),list(best[1][2:])


def _linear_exchange(F, ids, B, criterion, max_iter, *, _free_positions=None, max_elements=1_000_000):
    p = F.shape[1]
    history = []
    for iteration in range(max_iter+1):
        M, phi, diagonal, incoming, ratio, delta = _linear_scores(F,ids,B,criterion)
        condition = torch.linalg.cond(M).item()
        if iteration == max_iter:
            reason = "MAX_ITER"
            break
        tolerance = _tolerance(phi,phi,condition,condition,p)
        legal = (ratio>0)&(delta < -tolerance)
        if _free_positions is not None:
            legal &= _free_positions[:,None]
        moves = legal.flatten().nonzero().flatten()
        if len(moves):
            candidates = (([flat//len(incoming)],[incoming[flat%len(incoming)].item()])
                          for flat in moves[delta.flatten()[moves].argsort(stable=True)].tolist())
            reason = "NUMERICAL_INCONCLUSIVE"
        else:
            candidates = _linear_pair_moves(F,ids,criterion,max_elements,_free_positions,tolerance)
            reason = "LOCAL_2_NUMERICAL"
        for positions,incoming_ids in candidates:
            reason = "NUMERICAL_INCONCLUSIVE"
            trial = ids.clone()
            trial[positions] = trial.new_tensor(incoming_ids)
            X = F[trial]
            candidate = X.mT@X
            L, info = torch.linalg.cholesky_ex(candidate)
            if info.item():
                continue
            value = torch.trace(torch.cholesky_solve(B,L)).item()
            if phi-value > _tolerance(phi,value,condition,torch.linalg.cond(candidate).item(),p):
                history.append(dict(k=len(positions),outgoing_ids=ids[positions].tolist(),incoming_ids=incoming_ids))
                ids = trial
                break
        else:
            break
    return dict(ids=ids,objective=-phi,efficiency_lower_bound=phi/(len(ids)*diagonal.max().item()),
                stop_reason=reason,certificate="LOCAL_2_NUMERICAL" if reason=="LOCAL_2_NUMERICAL" else "NOT_CERTIFIED",swap_history=history)


def _block_information(F, ids, blocks):
    rows = F[ids].split(blocks)
    means = torch.stack([block.mean(0) for block in rows])
    centered = torch.cat([block - mean for block, mean in zip(rows, means)])
    return centered.mT @ centered, means


def _block_ai_gain(u, v, uq, vq, n11, n12, n22):
    ratio = n12.square() - n11*n22
    h11, h12, h22 = (u*uq).sum(1), u @ vq, (v*vq).sum()
    gain = (2*n12*h12 - n22*h11 - n11*h22) / ratio.masked_fill(ratio <= 0, torch.inf)
    return gain, ratio


def _whitened_moment(L, B):
    # In whitened coordinates H uses inv(L) @ B @ inv(L.T).
    Q = torch.linalg.solve_triangular(L, B, upper=False)
    return torch.linalg.solve_triangular(L, Q.mT, upper=False).mT


def _block_scores(F, ids, blocks, means, L, B=None, n_fixed=0):
    Z = torch.linalg.solve_triangular(L, torch.cat((F, means)).mT, upper=False).mT
    rows, centers = Z[:len(F)], Z[len(F):]
    if B is not None:
        T = Z @ _whitened_moment(L, B)
        weighted_rows, weighted_centers = T[:len(F)], T[len(F):]
    unused = torch.ones(len(F), dtype=torch.bool)
    unused[ids] = False
    incoming = unused.nonzero().flatten()
    groups = torch.arange(len(ids)).split(blocks)
    free_groups = [group[group >= n_fixed] for group in groups] if n_fixed else groups
    # Both rank-two kernels have det(K) = -1, so the ratio is 1 + tr(KG) - det(G).
    if len(incoming):
        for b, (group, center, size) in enumerate(zip(free_groups, centers, blocks)):
            if not len(group):
                continue
            u = rows[incoming] - center
            uu = u.square().sum(1)
            if B is not None:
                uq = weighted_rows[incoming] - weighted_centers[b]
            for i in group.tolist():
                v = rows[ids[i]] - center
                uv, vv = u @ v, v.square().sum()
                if B is None:
                    ratio = 1 + (1 - 1/size)*uu + (2/size)*uv - (1 + 1/size)*vv - (uu*vv - uv.square())
                    yield i, incoming, ratio.clamp_min(0).log(), False
                else:
                    vq = weighted_rows[ids[i]] - weighted_centers[b]
                    gain, ratio = _block_ai_gain(u, v, uq, vq, uu+1+1/size, uv+1/size, vv-1+1/size)
                    yield i, incoming, gain, False, ratio
    for r, outgoing in enumerate(free_groups):
        if not len(outgoing):
            continue
        for t in range(r + 1, len(groups)):
            other = free_groups[t]
            if not len(other):
                continue
            inverse_sizes = 1/blocks[r] + 1/blocks[t]
            e = centers[r] - centers[t]
            ee = e.square().sum()
            if B is not None:
                eq = weighted_centers[r] - weighted_centers[t]
            for i in outgoing.tolist():
                d = rows[ids[other]] - rows[ids[i]]
                dd, de = d.square().sum(1), d @ e
                if B is None:
                    ratio = 1 - inverse_sizes*dd - 2*de - (dd*ee - de.square())
                    yield i, other, ratio.clamp_min(0).log(), True
                else:
                    dq = weighted_rows[ids[other]] - weighted_rows[ids[i]]
                    gain, ratio = _block_ai_gain(d, e, dq, eq, dd, de-1, ee+1/blocks[r]+1/blocks[t])
                    yield i, other, gain, True, ratio


def _block_moves(F, ids, blocks, means, L, B, n_fixed):
    for i, choices, gains, interchange, *ratio in _block_scores(F, ids, blocks, means, L, B, n_fixed):
        improving = gains > 0
        if ratio:
            improving &= ratio[0] > 0
        for j in choices[improving].tolist():
            trial = ids.clone()
            trial[i] = ids[j] if interchange else j
            if interchange:
                trial[j] = ids[i]
            yield trial


def _plot_restarts(sizes, restarts, draw, information, moves, max_iter, layout, local_reason,
                   *, G, criterion="D", B=None):
    def attempt():
        start = ids = draw()
        if ids is None:
            return None
        M, summaries = information(ids)
        L, info = torch.linalg.cholesky_ex(M)
        pivot_floor = 64*len(M)*torch.finfo(M.dtype).eps*M.abs().amax()
        if not (info == 0 and (L.diagonal().square() > pivot_floor).all()):
            return None
        value = (torch.linalg.slogdet(M).logabsdet.item() if B is None
                 else -torch.trace(torch.cholesky_solve(B, L)).item())
        condition = torch.linalg.cond(M).item()
        for _ in range(max_iter):
            for trial in moves(ids, summaries, L):
                candidate, new_summaries = information(trial)
                if B is None:
                    sign, logdet = torch.linalg.slogdet(candidate)
                    if sign <= 0 or logdet <= value:
                        continue
                    new_value = logdet.item()
                else:
                    new_L, info = torch.linalg.cholesky_ex(candidate)
                    if info != 0:
                        continue
                    new_value = -torch.trace(torch.cholesky_solve(B, new_L)).item()
                    if new_value <= value:
                        continue
                new_condition = torch.linalg.cond(candidate).item()
                if not new_value - value > _tolerance(value, new_value, condition, new_condition, len(M)):
                    continue
                if B is None:
                    new_L, info = torch.linalg.cholesky_ex(candidate)
                    if info != 0:
                        continue
                ids, summaries, L = trial, new_summaries, new_L
                value, condition = new_value, new_condition
                break
            else:
                return start, (ids, value, local_reason), value
        return start, (ids, value, "MAX_ITER"), value

    (best_ids, best_value, best_reason), best_restart, starts, objectives = _restarts(restarts, attempt)
    certificate = best_reason if best_reason == local_reason else "NOT_CERTIFIED"
    ids = torch.cat([plot.sort().values for plot in best_ids.split(sizes)])
    M, _ = information(ids)
    floor = _dual_floor(G, M, sum(sizes), criterion, 0.0)
    return dict(ids=ids, **layout, objective=best_value, criterion=criterion,
                stop_reason=best_reason, certificate=certificate, efficiency_lower_bound=floor,
                restart_objectives=objectives, best_restart=best_restart, starts=starts)


def _valid_plot_sizes(sizes, n):
    return (isinstance(sizes, (list, tuple)) and sizes
            and all(type(size) is int and size > 0 for size in sizes)
            and type(n) is int and sum(sizes) == n)


def _plot_workspace(F, restarts, max_iter, max_elements, per_row, kind):
    _features(F, max_elements)
    integer(restarts, "restarts")
    integer(max_iter, "max_iter", 0)
    m, p = F.shape
    if 8*m*p + 8*p*p + per_row*m > max_elements:
        raise ValueError(f"insufficient max_elements for {kind} workspace")


def _covariate_design(F, n, Z, criterion, fixed_ids, restarts, generator, max_iter, max_elements):
    _features(F, max_elements)
    m, p = F.shape
    if (F.dtype != torch.float64 or F.device.type != "cpu" or criterion not in ("D", "A", "I")
            or not isinstance(Z, torch.Tensor) or Z.ndim != 2 or Z.shape[0] != n
            or Z.shape[1] < 1 or Z.dtype != F.dtype or Z.device != F.device or not torch.isfinite(Z).all()):
        raise ValueError("covariates requires CPU float64 F and finite (n, c) covariates, c >= 1, for D/A/I")
    if type(n) is not int or not m >= n >= 1 + Z.shape[1] + p:
        raise ValueError("covariates requires m >= n >= 1 + c + p")
    _plot_workspace(F, restarts, max_iter, max_elements - n*n - 4*n*(p+Z.shape[1]+1), 16, "covariate")
    X = torch.cat((F.new_ones(n, 1), Z), 1)
    gram = X.mT @ X
    L, info = torch.linalg.cholesky_ex(gram)
    if info != 0 or (L.diagonal().square() <= 64*len(gram)*torch.finfo(F.dtype).eps*gram.abs().amax()).any():
        raise ValueError("[1 covariates] must have full column rank")
    U = torch.linalg.qr(X, mode="reduced").Q
    Q = torch.eye(n, dtype=F.dtype) - U @ U.mT
    fixed, pool = (None, None) if fixed_ids is None else _fixed_pool(fixed_ids, m, n)
    n_fixed = 0 if fixed is None else len(fixed)
    centered = F - F.mean(0)
    B = None if criterion == "D" else torch.eye(p, dtype=F.dtype) if criterion == "A" else centered.mT @ centered / m

    def draw():
        if not n_fixed:
            return torch.randperm(m, generator=generator)[:n]
        return torch.cat((fixed, pool[torch.randperm(len(pool), generator=generator)[:n-n_fixed]]))

    def information(ids):
        R = Q @ F[ids]
        return R.mT @ R, R

    def moves(ids, R, L):
        rows, residuals = torch.linalg.solve_triangular(L, torch.cat((F, R)).mT, upper=False).mT.split((m, n))
        H = None if B is None else _whitened_moment(L, B)
        unused = torch.ones(m, dtype=torch.bool)
        unused[ids] = False
        incoming = unused.nonzero().flatten()
        # Exchange first, then interchange: both use the gate's det(K) = -1 kernel.
        for interchange in (False, True):
            for i in range(n_fixed, n):
                choices = torch.arange(i+1, n) if interchange else incoming
                d = rows[ids[choices] if interchange else choices] - rows[ids[i]]
                q = residuals[i] - residuals[choices] if interchange else residuals[i].expand_as(d)
                k = Q[i,i] + Q[choices,choices] - 2*Q[i,choices] if interchange else Q[i,i]
                a, b, e = d.square().sum(1), (d*q).sum(1), q.square().sum(1)
                ratio = 1 + k*a + 2*b - (a*e-b.square())
                if H is None:
                    improving = ratio > 1
                else:
                    dq, qq = d @ H, q @ H
                    gain = (2*(b+1)*(d*qq).sum(1) - (e-k)*(d*dq).sum(1) - a*(q*qq).sum(1)) / ratio.masked_fill(ratio <= 0, torch.inf)
                    improving = (ratio > 0) & (gain > 0)
                for j in choices[improving].tolist():
                    trial = ids.clone()
                    trial[i] = ids[j] if interchange else j
                    if interchange:
                        trial[j] = ids[i]
                    yield trial

    return _plot_restarts([1]*n, restarts, draw, information, moves, max_iter,
                          {"covariates": Z}, "LOCAL_COVARIATE_EXCHANGE", G=centered, criterion=criterion, B=B)


def _blocked_design(F, n, blocks, criterion, fixed_ids, restarts, generator, max_iter, max_elements):
    m, p = F.shape
    if not _valid_plot_sizes(blocks, n) or m < n or n - len(blocks) < p:
        raise ValueError("blocks must be positive Python ints summing to n, with m >= n and n - len(blocks) >= p")
    if criterion not in ("D", "A", "I"):
        raise ValueError("blocks supports criteria D, A and I")
    if F.dtype != torch.float64 or F.device.type != "cpu":
        raise ValueError("blocked search requires CPU float64 features")
    _plot_workspace(F, restarts, max_iter, max_elements, 16, f"blocked {criterion}")
    fixed, pool = (None, None) if fixed_ids is None else _fixed_pool(fixed_ids, m, n)
    n_fixed = 0 if fixed is None else len(fixed)

    def draw():
        if not n_fixed:
            return torch.randperm(m, generator=generator)[:n]
        return torch.cat((fixed, pool[torch.randperm(len(pool), generator=generator)[:n-n_fixed]]))

    centered = F - F.mean(0)
    B = None
    if criterion == "A":
        B = torch.eye(p, dtype=F.dtype)
    elif criterion == "I":
        B = centered.mT @ centered / m
    return _plot_restarts(
        blocks, restarts, draw,
        lambda ids: _block_information(F, ids, blocks),
        lambda ids, means, L: _block_moves(F, ids, blocks, means, L, B, n_fixed), max_iter,
        {"blocks": list(blocks)}, "LOCAL_BLOCK_EXCHANGE", G=centered, criterion=criterion, B=B)


def _split_information(F, ids, sizes, weights, blocks=None):
    rows = F[ids].split(sizes)
    sums = torch.stack([plot.sum(0) for plot in rows])
    # Center first to retain the small mean contribution at large eta.
    centered = torch.cat([plot - total / len(plot) for plot, total in zip(rows, sums)])
    M = centered.mT @ centered + sums.mT @ (weights[:, None] * sums)
    if blocks is not None:
        counts = F.new_tensor(sizes)
        w = weights * counts
        c = (1 - w) / counts
        first = 0
        shifted = []
        for size in blocks:
            last = first
            runs = 0
            while runs < size:
                runs += sizes[last]
                last += 1
            sl = slice(first, last)
            u = (w[sl, None] * sums[sl]).sum(0)
            tau = (w[sl] * counts[sl]).sum()
            M = M - torch.outer(u, u) / tau
            effective_c = c[sl] + w[sl].square() / tau
            shifted.append((c[sl, None] * sums[sl] + w[sl, None] * u / tau) / effective_c[:, None])
            first = last
        sums = torch.cat(shifted)
    return M, sums


def _split_ratios(incoming, outgoing, total, c):
    # The inputs are whitened by the current Cholesky factor, so G = W.T @ W.
    jj = incoming.square().sum(1)
    jo, js = incoming @ outgoing, incoming @ total
    oo, os, ss = (outgoing @ outgoing, outgoing @ total, total @ total)
    gram = torch.stack((jj, jo, js, jo, oo.expand_as(jj), os.expand_as(jj),
                        js, os.expand_as(jj), ss.expand_as(jj)), dim=1).reshape(-1, 3, 3)
    kernel = incoming.new_tensor([[1-c, c, -c], [c, -1-c, c], [-c, c, 0]])
    return torch.linalg.det(torch.eye(3, dtype=incoming.dtype) + kernel @ gram)


def _split_draw(hard, labels, sizes, generator, fixed, fixed_plots):
    unused = torch.ones(len(hard), dtype=torch.bool)
    if fixed_plots:
        unused[fixed] = False
    plots = []
    for k, size in enumerate(sizes):
        prefix = fixed_plots[k] if fixed_plots else None
        if prefix is not None and len(prefix):
            size -= len(prefix)
            plots.append(prefix)
            if not size:
                continue
            label = hard[prefix[0]]
        else:
            label = labels[torch.randint(len(labels), (1,), generator=generator)]
        pool = (unused & (hard == label)).nonzero().flatten()
        if len(pool) < size:
            return None
        plot = pool[torch.randperm(len(pool), generator=generator)[:size]]
        unused[plot] = False
        plots.append(plot)
    return torch.cat(plots)


def _split_ai_scores(d, q, dq, qq, c):
    return _block_ai_gain(d, q, dq, qq, d.square().sum(1), 1 + d @ q,
                          q.square().sum() - (1-c))


def _split_scores(F, ids, hard, groups, coefficients, sums, L, unused, B, n_fixed=0):
    rows, totals = torch.linalg.solve_triangular(
        L, torch.cat((F, sums)).mT, upper=False).mT.split((len(F), len(groups)))
    if B is not None:
        Q = _whitened_moment(L, B)
        weighted_rows, weighted_totals = rows @ Q, totals @ Q
    for k, (group, total, c) in enumerate(zip(groups, totals, coefficients)):
        pool = (unused & (hard == hard[ids[group[0]]])).nonzero().flatten()
        if n_fixed:
            group = group[group >= n_fixed]
            if not len(group):
                continue
        incoming = rows[pool]
        if B is not None:
            weighted_incoming = weighted_rows[pool]
        for pos in group.tolist():
            outgoing = ids[pos]
            if B is None:
                yield pos, pool, None, _split_ratios(incoming, rows[outgoing], total, c)
            else:
                gains, ratios = _split_ai_scores(
                    incoming - rows[outgoing], rows[outgoing] - c*total,
                    weighted_incoming - weighted_rows[outgoing],
                    weighted_rows[outgoing] - c*weighted_totals[k], c)
                yield pos, pool, gains, ratios


def _split_moves(F, ids, hard, labels, sizes, weights, coefficients, sums, L, B, n_fixed, blocks=None):
    unused = torch.ones(len(F), dtype=torch.bool)
    unused[ids] = False
    groups = torch.arange(len(ids)).split(sizes)
    for pos, pool, gains, ratios in _split_scores(F, ids, hard, groups, coefficients, sums, L, unused, B, n_fixed):
        improving = ratios > 1 if gains is None else (ratios > 0) & (gains > 0)
        for candidate in pool[improving].tolist():
            trial = ids.clone()
            trial[pos] = candidate
            yield trial
    for k, (group, c) in enumerate(zip(groups, coefficients)):
        if n_fixed and group[0] < n_fixed:
            continue
        for label in labels:
            if label == hard[ids[group[0]]]:
                continue
            pool = (unused & (hard == label)).nonzero().flatten()
            if len(pool) < len(group):
                continue
            trial, factor, total = ids.clone(), L, sums[k]
            for pos in group.tolist():
                whitened = torch.linalg.solve_triangular(
                    factor, torch.cat((F[pool], F[trial[pos:pos+1]], total[None])).mT,
                    upper=False).mT
                incoming, outgoing, whitened_total = whitened[:-2], whitened[-2], whitened[-1]
                if B is None:
                    ratios = _split_ratios(incoming, outgoing, whitened_total, c)
                    best = ratios.argmax().item()
                else:
                    d, q = incoming - outgoing, outgoing - c*whitened_total
                    Q = _whitened_moment(factor, B)
                    gains, ratios = _split_ai_scores(d, q, d @ Q, q @ Q, c)
                    valid = ratios > 0
                    if not valid.any():
                        break
                    best = gains.masked_fill(~valid, -torch.inf).argmax().item()
                if ratios[best] <= 0:
                    break
                trial[pos] = pool[best]
                pool = torch.cat((pool[:best], pool[best+1:]))
                M, trial_sums = _split_information(F, trial, sizes, weights, blocks)
                factor, info = torch.linalg.cholesky_ex(M)
                if info != 0:
                    break
                total = trial_sums[k]
            else:
                yield trial


def _split_design(F, n, sizes, hard, eta, criterion, fixed_ids, restarts, generator, max_iter, max_elements, blocks=None):
    if not isinstance(F, torch.Tensor) or F.ndim != 2 or F.dtype != torch.float64 or F.device.type != "cpu":
        raise ValueError(f"split-plot {criterion} search requires CPU float64 features")
    m, p = F.shape
    if not _valid_plot_sizes(sizes, n) or not m >= n >= p:
        raise ValueError("whole_plots must be positive Python ints summing to n, with m >= n >= p")
    if blocks is not None:
        boundaries = torch.tensor(sizes).cumsum(0).tolist()
        if (not _valid_plot_sizes(blocks, n) or n < len(blocks) + p
                or any(b not in boundaries for b in torch.tensor(blocks).cumsum(0).tolist())):
            raise ValueError("blocks must sum to n, align with plot boundaries and satisfy n >= len(blocks) + p")
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
                      *(getattr(torch, name, None) for name in ("uint16", "uint32", "uint64")))
    if (not isinstance(hard, torch.Tensor) or hard.shape != (m,)
            or hard.device != F.device or hard.dtype not in integer_dtypes):
        raise ValueError("hard must be a length-m integer tensor on F's device")
    if type(eta) not in (int, float) or not 0 < eta < float("inf"):
        raise ValueError("eta must be a finite positive Python int or float")
    _plot_workspace(F, restarts, max_iter, max_elements, 64, f"split-plot {criterion}")

    # Flip the high bit for uint64 to preserve unsigned label order in int64.
    hard = (hard.to(torch.int64) ^ -(1 << 63)
            if hard.dtype == getattr(torch, "uint64", None) else hard.to(torch.int64))
    labels = hard.unique()
    fixed, _ = (None, None) if fixed_ids is None else _fixed_pool(fixed_ids, m, n)
    n_fixed = 0 if fixed is None else len(fixed)
    fixed_plots = fixed.tensor_split(torch.tensor(sizes).cumsum(0)[:-1]) if n_fixed else ()
    if any(hard[prefix].unique().numel() > 1 for prefix in fixed_plots):
        raise ValueError("fixed_ids within a whole plot must share one hard label")
    counts = F.new_tensor(sizes)
    if eta >= 1:
        inverse_eta = 1 / eta
        coefficients = (counts + inverse_eta).reciprocal()
        weights = coefficients * inverse_eta / counts
    else:
        inverse_scale = (1 + eta * counts).reciprocal()
        coefficients = eta * inverse_scale
        weights = inverse_scale / counts
    if blocks is not None:
        w = weights * counts
        first = 0
        for size in blocks:
            last = first
            runs = 0
            while runs < size:
                runs += sizes[last]
                last += 1
            sl = slice(first, last)
            coefficients[sl] += w[sl].square() / (w[sl] * counts[sl]).sum()
            first = last
    coefficients = coefficients.tolist()
    G = F if blocks is None else F - F.mean(0)
    B = None
    if criterion == "A":
        B = torch.eye(p, dtype=F.dtype)
    elif criterion == "I":
        B = G.mT @ G / m

    return _plot_restarts(
        sizes, restarts, lambda: _split_draw(hard, labels, sizes, generator, fixed, fixed_plots),
        lambda ids: _split_information(F, ids, sizes, weights, blocks),
        lambda ids, sums, L: _split_moves(F, ids, hard, labels, sizes, weights, coefficients, sums, L, B, n_fixed, blocks),
        max_iter, dict(whole_plots=list(sizes), eta=eta, **({} if blocks is None else {"blocks": list(blocks)})), "LOCAL_SPLIT_PLOT_EXCHANGE",
        G=G, criterion=criterion, B=B)


@torch.no_grad()
def optimal_design(F: torch.Tensor, n: int, *, criterion: str = "D", fixed_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None, restarts: int = 10, generator: torch.Generator | None = None,
                   max_iter: int = 100, max_elements: int = 1_000_000, blocks: list[int] | tuple[int, ...] | None = None,
                   whole_plots: list[int] | tuple[int, ...] | None = None,
                   hard: torch.Tensor | None = None, eta: int | float = 1.0,
                   covariates: torch.Tensor | None = None) -> dict[str, object]:
    """D/A/I VND restarts with criterion-specific efficiency floor.

    Set blocks to fixed block sizes for CPU float64 D/A/I search without an intercept.
    Set whole_plots, hard labels and eta for CPU float64 split-plot D/A/I search.
    Set covariates to fixed per-run columns; ids keep run order and covariates never move.
    For structured designs, fixed_ids fill the leading positions, block by block.
    """
    if covariates is not None:
        if blocks is not None or whole_plots is not None:
            raise ValueError("covariates cannot be combined with blocks or whole_plots")
        if hard is not None:
            raise ValueError("hard requires whole_plots")
        return _covariate_design(F, n, covariates, criterion, fixed_ids, restarts, generator, max_iter, max_elements)
    if (whole_plots is None) != (hard is None):
        raise ValueError("whole_plots and hard must be given together")
    if whole_plots is not None:
        if criterion not in ("D", "A", "I"):
            raise ValueError("whole_plots supports criteria D, A and I")
        return _split_design(F, n, whole_plots, hard, eta, criterion, fixed_ids, restarts, generator, max_iter, max_elements, blocks)
    if blocks is not None:
        return _blocked_design(F,n,blocks,criterion,fixed_ids,restarts,generator,max_iter,max_elements)
    if criterion not in ("D","A","I"):
        raise ValueError('criterion must be D, A or I')
    if criterion == "D":
        result = d_optimal_design(F,n,fixed_ids=fixed_ids,restarts=restarts,generator=generator,
                                  max_iter=max_iter,max_elements=max_elements)
        return dict(result,criterion="D",efficiency_lower_bound=result['d_efficiency_lower_bound'])
    return _exact_design(F, n, criterion, fixed_ids, restarts, generator, max_iter, max_elements)


def _exact_design(F, n, criterion, fixed_ids, restarts, generator, max_iter, max_elements, tile_size=None):
    m, p = F.shape
    if type(n) is not int or not m >= n >= p:
        raise ValueError("n must be an int with m>=n>=p")
    if criterion == "D":
        _features(F,max_elements)
    fixed, pool = (None, None) if fixed_ids is None else _fixed_pool(fixed_ids,m,n)
    if criterion != "D":
        # Conservative simultaneous solve, gather and n-by-m score workspace; no m-by-m matrices.
        if 8*m*p+16*n*m+8*p*p > max_elements:
            raise ValueError('insufficient max_elements for linear-criterion workspace')
        B = torch.eye(p,dtype=F.dtype) if criterion=="A" else F.mT@F/m

    def attempt():
        if fixed is None:
            ids = torch.randperm(m, generator=generator)[:n].sort().values
        else:
            ids = torch.cat((fixed, pool[torch.randperm(len(pool), generator=generator)[:n-len(fixed)]])).sort().values
        if criterion == "D":
            if _verified_information(F,ids) is None:
                return None
        else:
            M = exact_information(F,ids,max_elements=max_elements)
            try:
                d_optimality(M,max_elements=max_elements)
            except ValueError:
                return None
        free = None if fixed is None else ~torch.isin(ids,fixed)
        if criterion == "D":
            result = k2_exchange(F, ids, max_iter=max_iter, tile_size=tile_size,
                                 max_elements=max_elements, _free_positions=free)
            return ids, result, result['final_objective']
        result = _linear_exchange(F, ids, B, criterion, max_iter, _free_positions=free, max_elements=max_elements)
        return ids, result, result['objective']

    best, best_restart, starts, objectives = _restarts(restarts, attempt)
    rows = F[best['ids']]
    current = (torch.tensor(-best['efficiency_gap_bound']/p,dtype=torch.float64).exp().item()
               if criterion == "D" else best['efficiency_lower_bound'])
    floor = _dual_floor(F,rows.mT@rows,n,criterion,current)
    if criterion == "D":
        best = dict(ids=best['ids'], objective=best['final_objective'], efficiency_gap_bound=best['efficiency_gap_bound'],
                    d_efficiency_lower_bound=floor, stop_reason=best['stop_reason'], certificate=best['certificate'])
    else:
        best['efficiency_lower_bound'] = floor
        best['criterion'] = criterion
    return dict(best,restart_objectives=objectives,best_restart=best_restart,starts=starts)


def next_batch(F: torch.Tensor, run_ids: list[int] | tuple[int, ...] | torch.Tensor, k: int, *,
               criterion: str = "D", restarts: int = 10, generator: torch.Generator | None = None,
               **kwargs: int) -> torch.Tensor:
    """Return sorted new int64 IDs by augmenting run_ids.

    Certificates refer to the full augmented design conditional on run_ids,
    for the current linearization only, rather than the new points alone.
    """
    ids = optimal_design(F, len(run_ids)+k, criterion=criterion, fixed_ids=run_ids if len(run_ids) else None,
                         restarts=restarts, generator=generator, **kwargs)["ids"]
    return ids[~torch.isin(ids, torch.as_tensor(run_ids, dtype=torch.int64))].sort().values
