"""Experimental streamed G-gather ratios; deliberately not a production export."""
import torch


def _unrank(ranks, size):
    i = torch.arange(size, device=ranks.device, dtype=torch.int64)
    prefix = i * (2*size-i-1) // 2
    first = torch.searchsorted(prefix[1:].contiguous(), ranks, right=True)
    second = first + 1 + ranks - prefix[first]
    return torch.stack((first, second), dim=1)


class _Scorer:
    """Task-local state/iterator; float32 is available only to the OBSERVED runner."""
    @torch.no_grad()
    def __init__(self, F, ids, *, tile_size=16384, max_elements=8_000_000,
                 form="gather4", observed_float32=False):
        allowed = (torch.float64, torch.float32) if observed_float32 else (torch.float64,)
        if not isinstance(F, torch.Tensor) or F.dtype not in allowed:
            raise TypeError("experimental public kernel requires float64 features")
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64:
            raise TypeError("ids must be int64")
        if F.ndim != 2 or ids.ndim != 1 or ids.device != F.device:
            raise ValueError("require F(m,p), matched-device ids(n), no batching")
        if form not in ("gather4", "add_first"):
            raise ValueError("form must be gather4 or add_first; delete-first prohibited")
        if type(tile_size) is not int or tile_size < 1 or type(max_elements) is not int or max_elements < 1:
            raise ValueError("positive Python int tile_size/max_elements required")
        m, p = F.shape
        n = ids.numel()
        if not m >= n >= p >= 1:
            raise ValueError("require m>=n>=p>=1")
        q = m-n
        fixed = 8*m*p + 6*p*p + 4*n*n + 4*n*q + 6*m + 64
        available = max_elements-fixed
        if m*p > max_elements or fixed > max_elements or available < 256 + 2*q:
            raise ValueError("insufficient max_elements before numerical allocation")
        if not bool(torch.isfinite(F).all()) or bool(((ids < 0) | (ids >= m)).any()):
            raise ValueError("nonfinite F or out-of-range IDs")
        ordered = ids.sort().values
        if bool((ordered[1:] == ordered[:-1]).any()):
            raise ValueError("duplicate IDs")
        X = F[ids]
        M = X.mT@X
        if not bool(torch.isfinite(M).all()):
            raise ValueError("initial Gram overflow")
        L, info = torch.linalg.cholesky_ex(M)
        tau = 64*p*torch.finfo(F.dtype).eps*M.abs().max()
        if bool(info != 0) or bool((L.diagonal().square() <= tau).any()):
            raise ValueError("initial Gram fails numerical SPD/pivot policy")
        self.condition = torch.linalg.cond(M).item()
        self.F, self.ids, self.form = F, ids, form
        self.n, self.q, self.p = n, q, p
        self.count = (n*(n-1)//2)*(q*(q-1)//2)
        mask = torch.ones(m, dtype=torch.bool, device=F.device)
        mask[ids] = False
        self.pool = mask.nonzero().flatten()
        Z = torch.linalg.solve_triangular(L, F.mT, upper=False)
        R, self.U = Z[:, ids], Z[:, self.pool]
        self.ss, self.sn = R.mT@R, R.mT@self.U
        # Full G_NN only when it fits; otherwise lazily cache bounded row blocks.
        self.nn = self.U.mT@self.U if q*q <= available-256 else None
        self.diag = self.nn.diagonal() if self.nn is not None else self.U.square().sum(0)
        reserve = q*q if self.nn is not None else 2*q
        self.width = min(tile_size, (available-reserve)//256)
        self.row_height = 0 if self.nn is not None else max(1, min(q, (available-256*self.width)//max(1,2*q)))
        reserve = q*q if self.nn is not None else 2*q*self.row_height
        self.workspace_elements = fixed+reserve+256*self.width
        self._cache_key, self._cache = None, None

    def _nn_values(self, a, b):
        if self.nn is not None:
            return self.nn[a, b]
        result = torch.empty(a.shape, dtype=self.F.dtype, device=self.F.device)
        blocks = a // self.row_height
        for key in blocks.unique().tolist():
            lo, hi = key*self.row_height, min(self.q, (key+1)*self.row_height)
            if key != self._cache_key:
                self._cache = self.U[:, lo:hi].mT@self.U
                self._cache_key = key
            mask = blocks == key
            result[mask] = self._cache[a[mask]-lo, b[mask]]
        return result

    @torch.no_grad()
    def evaluate(self, ranks):
        if ranks.numel() > self.width:
            raise ValueError("rank request exceeds effective tile width")
        incoming_count = self.q*(self.q-1)//2
        outgoing = _unrank(ranks//incoming_count, self.n)
        new = _unrank(ranks % incoming_count, self.q)
        oo = self.ss[outgoing[:, :, None], outgoing[:, None, :]]
        on = self.sn[outgoing[:, :, None], new[:, None, :]]
        nn = torch.empty_like(oo)
        nn[:, 0, 0], nn[:, 1, 1] = self.diag[new[:, 0]], self.diag[new[:, 1]]
        nn[:, 0, 1] = self._nn_values(new[:, 0], new[:, 1])
        nn[:, 1, 0] = self._nn_values(new[:, 1], new[:, 0])
        if self.form == "gather4":
            H = torch.cat((torch.cat((oo, on), dim=2), torch.cat((on.mT, nn), dim=2)), dim=1)
            signs = self.F.new_tensor([-1., -1., 1., 1.])
            ratios = torch.linalg.det(torch.eye(4, dtype=self.F.dtype, device=self.F.device)+signs[None, :, None]*H)
        else:
            A = nn+torch.eye(2, dtype=self.F.dtype, device=self.F.device)
            a,b,c,d = A[:,0,0],A[:,0,1],A[:,1,0],A[:,1,1]
            determinant = a*d-b*c
            Y = on.mT
            # Closed-form two-row solve, not an explicit inverse matrix.
            solution = torch.stack(((d[:,None]*Y[:,0]-b[:,None]*Y[:,1])/determinant[:,None],
                                    (a[:,None]*Y[:,1]-c[:,None]*Y[:,0])/determinant[:,None]), dim=1)
            S = torch.eye(2, dtype=self.F.dtype, device=self.F.device)-oo+on@solution
            ratios = determinant*(S[:,0,0]*S[:,1,1]-S[:,0,1]*S[:,1,0])
        return outgoing, self.pool[new], ratios

    def tiles(self):
        for start in range(0, self.count, self.width):
            ranks = torch.arange(start, min(start+self.width,self.count), device=self.F.device)
            positions, incoming, ratios = self.evaluate(ranks)
            yield start, positions, incoming, ratios


def gather_two_swap_ratios(F, ids, *, tile_size=16384, max_elements=8_000_000, form="gather4"):
    """Return an ordered iterator of ratio tensors, not an O(pair-count) result.

    Eager validation; no SPD conclusion about prospective swaps, winner or search.
    """
    state = _Scorer(F, ids, tile_size=tile_size, max_elements=max_elements, form=form)
    return (ratios for _, _, _, ratios in state.tiles())
