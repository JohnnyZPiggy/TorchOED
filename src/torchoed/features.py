from collections.abc import Sequence

import torch

from ._validation import budget, floating, integer


def jacobian_features(model: torch.nn.Module, X: torch.Tensor, *, params: Sequence[str] | None = None, max_elements: int = 1_000_000) -> torch.Tensor:
    """Detached scalar-response Jacobian rows at the module's current parameters.

    Call model.eval() first: row-coupling layers otherwise differentiate the
    coupled model(X). Certificates apply only to this linearized problem over
    the supplied candidates, not the nonlinear model at its true parameters.
    By default use all requires_grad parameters in named_parameters() order;
    explicit names select and order flattened parameter columns. The module
    is unchanged. Output uses the model's parameter dtype; m*q elements are
    budgeted before differentiation. jacrev takes one reverse pass per row
    (vectorized), O(m*cost(f)) time and O(m*q) output memory.
    """
    named = dict(model.named_parameters())
    names = [name for name, p in named.items() if p.requires_grad] if params is None else list(params)
    selected = {name: named[name] for name in names}
    if not selected:
        raise ValueError("select at least one parameter")
    m = X.shape[0]
    budget((m, sum(p.numel() for p in selected.values())), max_elements)
    # functional_call reflects in-place buffer updates; isolate module state.
    state = {name: p.detach().clone() for name, p in named.items()}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    selected = {name: state[name] for name in selected}

    def response(values: dict[str, torch.Tensor]) -> torch.Tensor:
        y = torch.func.functional_call(model, (state | values, buffers), (X,))
        if y.shape not in ((m,), (m, 1)):
            raise ValueError("model(X) must have shape (m,) or (m,1)")
        return y.reshape(m)

    derivatives = torch.func.jacrev(response)(selected)
    return torch.cat([d.reshape(m, -1) for d in derivatives.values()], dim=1).detach()


def polynomial_features(C: torch.Tensor, degree: int = 2, include_bias: bool = True, *, max_elements: int = 1_000_000) -> torch.Tensor:
    """[bias, linear, squares, pair products i<j]; preserves autograd."""
    if not isinstance(C, torch.Tensor):
        raise TypeError("C must be a floating torch.Tensor")
    floating(C.dtype)
    if C.ndim not in (2, 3) or C.shape[-1] < 1:
        raise ValueError("C must have shape (m,k) or (B,m,k), k>=1")
    if type(degree) is not int or degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if type(include_bias) is not bool:
        raise TypeError("include_bias must be bool")
    k = C.shape[-1]
    p = int(include_bias) + k + (k + k * (k - 1) // 2 if degree == 2 else 0)
    budget((*C.shape[:-1], p), max_elements)
    if not bool(torch.isfinite(C).all()):
        raise ValueError("C must contain only finite values")
    parts = [torch.ones_like(C[..., :1])] if include_bias else []
    parts.append(C)
    if degree == 2:
        i, j = torch.triu_indices(k, k, offset=1, device=C.device)
        parts.extend((C.square(), C[..., i] * C[..., j]))
    return torch.cat(parts, dim=-1)


def effects_coding(codes: torch.Tensor, levels: int) -> torch.Tensor:
    integer(levels, 'levels', 2)
    if codes.dtype != torch.int64 or bool(((codes<0)|(codes>=levels)).any()):
        raise ValueError('codes must be int64 in 0..levels-1')
    return (codes[:,None]==torch.arange(levels-1,device=codes.device)).double()-(codes==levels-1)[:,None].double()


def feasible_candidates(C: torch.Tensor, *, A: torch.Tensor | None = None, b: torch.Tensor | None = None, disallowed: list[dict[int, float]] | tuple[dict[int, float], ...] = ()) -> torch.Tensor:
    mask = torch.ones(len(C),dtype=torch.bool,device=C.device)
    if A is not None:
        mask &= (C@A.mT <= b).all(dim=-1)
    for rule in disallowed:
        forbidden = torch.ones_like(mask)
        for column,value in rule.items():
            forbidden &= C[:,column]==value
        mask &= ~forbidden
    return mask
