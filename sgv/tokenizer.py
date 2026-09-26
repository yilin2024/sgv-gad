"""Graph tokenizer: the same label-free computation on every graph.

Per graph it produces
    F      n x z_dim   feature block: PCA of the column-scaled attributes (uncentred projection), columns ordered
                       from smoothest to roughest on the graph, signs fixed; raw attributes zero-padded when d <= z_dim
    omega  n x z_dim   Spectral Graph Vocabulary: log(1 + N * share of the local Dirichlet energy) of every column of F
    joint  n x 9       rotation-invariant energy scalars of the whole column block
    phi    n x C       band fractions beta_1..beta_C averaged over columns (router warm-up target)
    zeta   n x 8       random-walk return profile
    logdeg n           log(1 + degree)

Local energy of a column f, with v = M^{-1/2} f (0 on isolated nodes):
    delta_i = v_i^2 + sum_{j in N(i)} v_j^2
    eps_i   = sum_{j in N(i)} (v_i - v_j)^2,       chi_i = eps_i / delta_i
    omega_i = log(1 + eps_i / mean_k delta_k)      node i's share of the column's local energy, times N
"""
from __future__ import annotations

import os
import zlib
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .config import Config, TokenizerCfg
from .data import Graph
from .graph_ops import GraphOps


# ------------------------------------------------------------------------------------------------ feature block F
def rank_basis(Xc: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U (n x r), sigma (r,), V (d x r) of Xc, sigma descending and > tol * sigma_max; via the smaller Gram."""
    n, d = Xc.shape
    if d <= n:
        w, V = np.linalg.eigh(Xc.T @ Xc)
        keep = w > (tol ** 2) * w.max()
        w, V = w[keep], V[:, keep]
        sig = np.sqrt(w)
        U = (Xc @ V) / sig
    else:
        w, U = np.linalg.eigh(Xc @ Xc.T)
        keep = w > (tol ** 2) * w.max()
        w, U = w[keep], U[:, keep]
        sig = np.sqrt(w)
        V = (Xc.T @ U) / sig
    order = np.argsort(-sig)
    return U[:, order], sig[order], V[:, order]


def smoothness_order(F: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    """Column order by Dirichlet energy of the min-max scaled column, ascending (smoothest first)."""
    lo, hi = F.min(0), F.max(0)
    Xs = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    deg = np.asarray(A.sum(1)).ravel()
    energy = (deg[:, None] * Xs * Xs).sum(0) - np.einsum("ij,ij->j", Xs, A @ Xs)
    return np.argsort(energy)


def fix_signs(F: np.ndarray) -> np.ndarray:
    """Flip each column so that its largest-magnitude entry is positive."""
    idx = np.abs(F).argmax(0)
    s = np.sign(F[idx, np.arange(F.shape[1])])
    s[s == 0] = 1.0
    return F * s


def feature_block(X: np.ndarray, A: sp.csr_matrix, tc: TokenizerCfg) -> Tuple[np.ndarray, np.ndarray]:
    """(F, cols): F is n x z_dim float32; cols are the columns the energies are computed on (never the padding)."""
    X = np.asarray(X, dtype=np.float64)
    std = X.std(0)
    keep = std > 0
    X, std = X[:, keep], std[keep]
    n, d = X.shape
    F = np.zeros((n, tc.z_dim))
    if d <= tc.z_dim:
        F[:, :d] = X
        cols = X.copy()
    else:
        Xs = X / std
        _, _, V = rank_basis(Xs - Xs.mean(0), tc.rank_tol)
        Fr = Xs @ V[:, :tc.z_dim]
        Fr = fix_signs(Fr[:, smoothness_order(Fr, A)])
        F[:, :Fr.shape[1]] = Fr
        cols = Fr
    return F.astype(np.float32), cols


# ------------------------------------------------------------------------------------------------ local energies
def apply_khop(ops: GraphOps, X: torch.Tensor, k: int) -> torch.Tensor:
    for _ in range(k):
        X = ops.A_mm(X)
    return X


def energy_ratio(ops: GraphOps, f: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """(chi, delta), n x c, over the k-step walk neighbourhood A^k (k = 1: the plain neighbourhood)."""
    v = ops.dinv_sqrt[:, None] * f
    vv = v * v
    Mv, Mvv = apply_khop(ops, v, k), apply_khop(ops, vv, k)
    mk = apply_khop(ops, torch.ones(ops.n, 1, dtype=f.dtype, device=f.device), k)
    num = (mk * vv - 2.0 * v * Mv + Mvv).clamp_min(0.0)
    delta = vv + Mvv
    chi = torch.where(delta > 0, num / delta.clamp_min(torch.finfo(f.dtype).tiny), torch.zeros_like(num))
    return chi, delta


def weighted_chi(chi: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """eps_i / mean_k delta_k = N * chi_i * delta_i / sum_k delta_k per column; 0 where the column has no energy."""
    m = delta.mean(0, keepdim=True)
    return torch.where(m > 0, chi * delta / m.clamp_min(torch.finfo(delta.dtype).tiny), torch.zeros_like(chi))


def vocabulary(ops: GraphOps, cols: torch.Tensor, z_dim: int) -> torch.Tensor:
    """(n, z_dim) Spectral Graph Vocabulary log(1 + weighted chi) of every column, in column order, zero-padded."""
    chi, delta = energy_ratio(ops, cols, 1)
    w = torch.log1p(weighted_chi(chi, delta))
    out = torch.zeros(w.shape[0], z_dim, dtype=w.dtype, device=w.device)
    out[:, :w.shape[1]] = w[:, :z_dim]
    return out


def band_fractions(ops: GraphOps, f: torch.Tensor, C: int) -> torch.Tensor:
    """(C+1, n, c) share of each Beta band in the per-column energy; sums to 1 over the first axis."""
    e = ops.beta_bank(f, C) ** 2
    s = e.sum(0, keepdim=True)
    return torch.where(s > 0, e / s.clamp_min(torch.finfo(f.dtype).tiny), torch.full_like(e, 1.0 / (C + 1)))


def joint_energy(ops: GraphOps, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Energy ratio on the row vectors of a column block: invariant to any orthogonal change of attribute basis."""
    v = ops.dinv_sqrt[:, None] * F
    sq = (v * v).sum(1)
    Msq = ops.A_mm(sq[:, None])[:, 0]
    mk = ops.A_mm(torch.ones(ops.n, 1, dtype=F.dtype, device=F.device))[:, 0]
    num = (mk * sq - 2.0 * (v * ops.A_mm(v)).sum(1) + Msq).clamp_min(0.0)
    den = sq + Msq
    chi = torch.where(den > 0, num / den.clamp_min(torch.finfo(F.dtype).tiny), torch.zeros_like(num))
    return chi, den


def joint_scalars(ops: GraphOps, cols: torch.Tensor, tc: TokenizerCfg) -> torch.Tensor:
    """(n, 4 + C) = [log(1+chi) at hops 0, 1, 2 | band fractions beta_0..beta_C | log delta] of the whole block."""
    chi0, delta0 = joint_energy(ops, cols)
    f1 = ops.P_mm(cols)
    f2 = ops.P_mm(f1)
    out = [torch.log1p(chi0), torch.log1p(joint_energy(ops, f1)[0]), torch.log1p(joint_energy(ops, f2)[0])]
    e = (ops.beta_bank(cols, tc.beta_order) ** 2).sum(-1)
    t = e.sum(0, keepdim=True)
    out += list(torch.where(t > 0, e / t.clamp_min(torch.finfo(cols.dtype).tiny),
                            torch.full_like(e, 1.0 / (tc.beta_order + 1))))
    out.append(torch.log(delta0 + tc.log_delta_eps))
    return torch.stack(out, 1)


def probe_seed(name: str, base: int) -> int:
    """Fixed per graph, so the same graph always gets the same probes."""
    return (base * 1_000_003 + zlib.crc32(name.encode())) % (2 ** 31)


def rw_return_profile(ops: GraphOps, steps: int, probes: int, seed: int) -> torch.Tensor:
    """zeta_i = (R^k)_ii for k = 1..steps; k = 2 exact, k >= 3 by a Hutchinson estimate with Rademacher probes."""
    out = torch.zeros(ops.n, steps, dtype=ops.dtype, device=ops.device)
    if steps >= 2:
        out[:, 1] = ops.rinv * ops.A_mm(ops.rinv[:, None])[:, 0]
    if steps >= 3:
        g = torch.Generator(device="cpu").manual_seed(seed)
        Z = (torch.randint(0, 2, (ops.n, probes), generator=g) * 2 - 1).to(ops.dtype).to(ops.device)
        W = Z
        for k in range(1, steps + 1):
            W = ops.RW_mm(W)
            if k >= 3:
                out[:, k - 1] = (Z * W).mean(1)
    return out


# ------------------------------------------------------------------------------------------------ entry point
def tokenize(g: Graph, cfg: Config) -> Dict[str, np.ndarray]:
    tc = cfg.tokenizer
    device = tc.device if (tc.device == "cpu" or torch.cuda.is_available()) else "cpu"
    F, cols_np = feature_block(g.X, g.A, tc)
    ops = GraphOps(g.A, device=device, dtype=torch.float32)
    cols = torch.as_tensor(cols_np, dtype=torch.float32, device=device)
    phi = band_fractions(ops, cols, tc.beta_order).mean(2).T[:, 1:]            # (n, C): beta_1..beta_C
    out = dict(F=F,
               omega=vocabulary(ops, cols, tc.z_dim),
               joint=joint_scalars(ops, cols, tc),
               phi=phi,
               zeta=rw_return_profile(ops, tc.rwpe_steps, tc.rwpe_probes, probe_seed(g.name, tc.rwpe_seed)))
    out = {k: (v.cpu().numpy() if torch.is_tensor(v) else v).astype(np.float32) for k, v in out.items()}
    out["logdeg"] = np.log1p(g.m).astype(np.float32)
    out["y"] = g.y.astype(np.int8)
    for k, v in out.items():
        assert np.isfinite(v).all(), f"{g.name}: non-finite {k}"
    return out


def load_or_tokenize(g: Graph, cfg: Config, cache_dir: str | None) -> Dict[str, np.ndarray]:
    """Tokens of one graph, cached as {cache_dir}/{name}_{tokenizer hash}.npz when cache_dir is given."""
    p = os.path.join(cache_dir, f"{g.name}_{cfg.hash('tokenizer')}.npz") if cache_dir else None
    if p and os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    tok = tokenize(g, cfg)
    if p:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(p, **tok)
    return tok
