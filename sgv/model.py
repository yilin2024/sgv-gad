"""Backbone: attention and routed spectral experts in parallel, then a direction head.

    Q^(0) = W_in x_i + W_pos p_i                  x_i = [F || omega || joint] (rank-normalised), p_i = [zeta || log d]
    block l = 1..L (pre-LN):
        Q~  = LN(Q)
        Q'  = Q + mu^(l) Att(Q~) + (1 - mu^(l)) Spec(Q~)
        Q'' = Q' + FFN(LN(Q'))
    Spec(Q~) = beta~_0(I - P) Q~ W_0 + sum_{nu=1..C} alpha_{., nu} * beta~_nu(I - P) Q~ W_nu
    alpha_i  = softmax(MLP(Q~_i) / kappa)          router, one per block
    a_i      = MLP(LN(Q^(L)_i))                    anomaly logit
Each spectral expert beta~_nu is a Chebyshev polynomial of order K whose coefficients are trainable and initialised
so that beta~_nu equals the Beta wavelet kernel beta_nu exactly.
"""
from __future__ import annotations

import dataclasses as dc
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from numpy.polynomial import chebyshev as npc
from numpy.polynomial import polynomial as npp
from torch.utils.checkpoint import checkpoint

from .config import ModelCfg
from .graph_ops import GraphOps, beta_coefficients


# ------------------------------------------------------------------------------------------------ spectral experts
def beta_to_cheb(C: int, K: int) -> np.ndarray:
    """(C+1, K+1): row nu holds the Chebyshev coefficients of beta_nu in the variable w - 1."""
    assert C <= K, "Chebyshev order must be at least the Beta order"
    mono = beta_coefficients(C)
    out = np.zeros((C + 1, K + 1))
    for nu in range(C + 1):
        shifted = npp.Polynomial(mono[nu])(npp.Polynomial([1.0, 1.0]))
        ch = npc.poly2cheb(shifted.coef)
        out[nu, :len(ch)] = ch
    return out


def cheb_basis(ops: GraphOps, X: torch.Tensor, K: int) -> torch.Tensor:
    """(K+1, n, h) stack of Ch_q(-P) X; -P = (I - P) - I has spectrum in [-1, 1]."""
    T = [X, -ops.P_mm(X)]
    for _ in range(2, K + 1):
        T.append(-2.0 * ops.P_mm(T[-1]) - T[-2])
    return torch.stack(T[:K + 1])


class SpectralBranch(nn.Module):
    def __init__(self, h: int, K: int, C: int):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(beta_to_cheb(C, K), dtype=torch.float32))
        self.W = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(C + 1)])
        self.C = C

    def forward(self, basis: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        filt = torch.einsum("vq,qnh->vnh", self.coef, basis)
        out = sum(alpha[:, nu - 1:nu] * self.W[nu](filt[nu]) for nu in range(1, self.C + 1))
        return out + self.W[0](filt[0])                   # beta~_0: shared low-pass expert


class Router(nn.Module):
    def __init__(self, d_in: int, hidden: int, C: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, C))

    def forward(self, r: torch.Tensor, kappa: float) -> torch.Tensor:
        return torch.softmax(self.mlp(r) / kappa, dim=-1)


# ------------------------------------------------------------------------------------------------ attention
@dc.dataclass
class Neighbourhood:
    crow: torch.Tensor                 # (n+1,) destination-major CSR offsets, self-edges included
    src: torch.Tensor
    dst: torch.Tensor                  # sorted
    _chunks: Optional[Tuple[int, List[Tuple[int, int, int, int]]]] = None

    def chunks(self, chunk_edges: int) -> List[Tuple[int, int, int, int]]:
        """(row0, row1, edge0, edge1) slices of about chunk_edges edges each; a row is never split."""
        if self._chunks is None or self._chunks[0] != chunk_edges:
            c = self.crow.cpu().numpy()
            n, E = c.size - 1, int(c[-1])
            cuts = np.searchsorted(c, np.arange(chunk_edges, E, chunk_edges), side="left")
            b = np.unique(np.r_[0, cuts, n])
            self._chunks = (chunk_edges, [(int(b[i]), int(b[i + 1]), int(c[b[i]]), int(c[b[i + 1]]))
                                          for i in range(len(b) - 1) if b[i + 1] > b[i]])
        return self._chunks[1]


def capped_neighbourhood(crow: torch.Tensor, col: torch.Tensor, cap: Optional[int],
                         gen: Optional[torch.Generator] = None) -> Neighbourhood:
    """At most `cap` uniformly sampled neighbours per node (all when cap is None), plus a self-edge."""
    dev = crow.device
    n = crow.numel() - 1
    deg = crow[1:] - crow[:-1]
    rows = torch.repeat_interleave(torch.arange(n, device=dev), deg)
    src = col
    if cap is not None and rows.numel() and int(deg.max()) > cap:
        key = rows.to(torch.float64) + torch.rand(rows.numel(), generator=gen, device=dev, dtype=torch.float64)
        order = torch.argsort(key)
        rank = torch.arange(rows.numel(), device=dev) - crow[:-1][rows]
        keep = order[rank < cap]
        rows, src = rows[keep], col[keep]
    selfe = torch.arange(n, device=dev)
    dst = torch.cat([rows, selfe])
    src = torch.cat([src, selfe])
    perm = torch.argsort(dst, stable=True)
    dst, src = dst[perm], src[perm]
    crow2 = torch.zeros(n + 1, dtype=torch.long, device=dev)
    crow2[1:] = torch.cumsum(torch.bincount(dst, minlength=n), 0)
    return Neighbourhood(crow2, src, dst)


class GraphAttention(nn.Module):
    """GAT-style multi-head attention over the neighbourhood and the node itself, processed in edge chunks."""

    def __init__(self, h: int, heads: int, dropout: float = 0.0, bf16: bool = True):
        super().__init__()
        self.h, self.H, self.dh = h, heads, h // heads
        self.W = nn.Linear(h, h, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * self.dh ** -0.5)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * self.dh ** -0.5)
        self.bias = nn.Parameter(torch.zeros(h))
        self.dropout, self.bf16 = dropout, bf16

    def _chunk(self, Wq, s_src, s_dst, src, dst, r0: int, r1: int) -> torch.Tensor:
        loc = dst - r0
        e = Fn.leaky_relu(s_src[src] + s_dst[dst], 0.2)
        with torch.no_grad():
            m = torch.full((r1 - r0, self.H), float("-inf"), device=e.device).scatter_reduce(
                0, loc[:, None].expand(-1, self.H), e, "amax", include_self=True)
        w = torch.exp(e - m[loc])
        den = torch.zeros(r1 - r0, self.H, device=e.device).index_add(0, loc, w)
        att = Fn.dropout(w / den[loc], self.dropout, self.training)
        msg = (att.to(Wq.dtype)[..., None] * Wq[src]).float()
        return torch.zeros(r1 - r0, self.H, self.dh, device=e.device).index_add(0, loc, msg)

    def forward(self, Qt: torch.Tensor, nbr: Neighbourhood, chunk_edges: int) -> torch.Tensor:
        n = Qt.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and Qt.is_cuda):
            Wq = self.W(Qt)
        Wq = Wq.view(n, self.H, self.dh)
        s_src = (Wq.float() * self.a_src).sum(-1)
        s_dst = (Wq.float() * self.a_dst).sum(-1)
        ckpt = self.training and torch.is_grad_enabled()
        outs = []
        for r0, r1, e0, e1 in nbr.chunks(chunk_edges):
            args = (Wq, s_src, s_dst, nbr.src[e0:e1], nbr.dst[e0:e1], r0, r1)
            outs.append(checkpoint(self._chunk, *args, use_reentrant=False) if ckpt else self._chunk(*args))
        return torch.cat(outs).reshape(n, self.h) + self.bias


# ------------------------------------------------------------------------------------------------ backbone
class Block(nn.Module):
    def __init__(self, mc: ModelCfg, C: int):
        super().__init__()
        h = mc.hidden
        self.mc = mc
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.att = GraphAttention(h, mc.heads, mc.dropout, mc.attn_bf16)
        self.spec = SpectralBranch(h, mc.cheb_order, C)
        self.router = Router(h, mc.router_hidden, C)
        self.mu = nn.Parameter(torch.tensor(0.5))
        self.ffn = nn.Sequential(nn.Linear(h, mc.ffn_mult * h), nn.GELU(), nn.Dropout(mc.dropout),
                                 nn.Linear(mc.ffn_mult * h, h), nn.Dropout(mc.dropout))

    def forward(self, Q: torch.Tensor, g, kappa: float):
        Qt = self.ln1(Q)
        upd = self.mu * self.att(Qt, g.nbr, self.mc.attn_chunk_edges)
        alpha = self.router(Qt, kappa)
        upd = upd + (1.0 - self.mu) * self.spec(cheb_basis(g.ops, Qt, self.mc.cheb_order), alpha)
        Q = Q + upd
        return Q + self.ffn(self.ln2(Q)), alpha


class Backbone(nn.Module):
    def __init__(self, mc: ModelCfg, d_in: int, d_pos: int, C: int):
        super().__init__()
        h = mc.hidden
        self.W_in = nn.Linear(d_in, h)
        self.W_pos = nn.Linear(d_pos, h)
        self.blocks = nn.ModuleList([Block(mc, C) for _ in range(mc.blocks)])
        self.ln_f = nn.LayerNorm(h)
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, g, kappa: float):
        Q = self.W_in(g.x) + self.W_pos(g.pos)
        alphas = []
        for b in self.blocks:
            Q, alpha = b(Q, g, kappa)
            alphas.append(alpha)
        return self.head(self.ln_f(Q)).squeeze(-1), alphas
