"""Sparse graph operators (torch CSR; nothing dense n x n is ever formed).

With the loop-free adjacency A and loop-free degree m_i:
    P     = M^{-1/2} A M^{-1/2}        normalised adjacency; rows and columns of isolated nodes are zero
    I - P                              normalised Laplacian, spectrum in [0, 2]
    R     = M^{-1} A                   random walk without self-loops
    beta_nu(w) = (w/2)^nu (1 - w/2)^(C-nu) / (2 B(nu+1, C+1-nu)),  nu = 0..C,  applied as beta_nu(I - P)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import beta as beta_fn


class _SymSpMM(torch.autograd.Function):
    """M @ X for a symmetric sparse M; the backward is M @ grad, so no CSR transpose is needed."""

    @staticmethod
    def forward(ctx, M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ctx.M = M
        return M @ X

    @staticmethod
    def backward(ctx, G: torch.Tensor):
        return None, ctx.M @ G.contiguous()


def beta_coefficients(C: int) -> np.ndarray:
    """(C+1) x (C+1) array; row nu holds c_{nu,k} with beta_nu(w) = sum_k c_{nu,k} w^k."""
    out = np.zeros((C + 1, C + 1))
    for nu in range(C + 1):
        p = (np.polynomial.Polynomial([0.0, 0.5]) ** nu) * (np.polynomial.Polynomial([1.0, -0.5]) ** (C - nu))
        c = p.coef / (2.0 * beta_fn(nu + 1, C + 1 - nu))
        out[nu, :len(c)] = c
    return out


class GraphOps:
    """A, P and R as torch CSR tensors sharing one sparsity pattern."""

    def __init__(self, A: sp.csr_matrix, device: str = "cpu", dtype: torch.dtype = torch.float32):
        A = sp.csr_matrix(A)
        A.sort_indices()
        n = A.shape[0]
        m = np.asarray(A.sum(1)).ravel()
        s = np.where(m > 0, 1.0 / np.sqrt(np.maximum(m, 1.0)), 0.0)
        r = np.where(m > 0, 1.0 / np.maximum(m, 1.0), 0.0)
        rows = np.repeat(np.arange(n), np.diff(A.indptr))
        col = A.indices.astype(np.int64)
        crow = torch.as_tensor(A.indptr.astype(np.int64), device=device)
        colt = torch.as_tensor(col, device=device)

        def csr(vals: np.ndarray) -> torch.Tensor:
            return torch.sparse_csr_tensor(crow, colt, torch.as_tensor(vals, dtype=dtype, device=device),
                                           size=(n, n))

        self.n, self.device, self.dtype = n, device, dtype
        self.m = torch.as_tensor(m, dtype=dtype, device=device)
        self.dinv_sqrt = torch.as_tensor(s, dtype=dtype, device=device)
        self.rinv = torch.as_tensor(r, dtype=dtype, device=device)
        self._A = csr(np.ones(col.size))
        self._P = csr(s[rows] * s[col])
        self._RW = csr(r[rows])

    def A_mm(self, X: torch.Tensor) -> torch.Tensor:
        return _SymSpMM.apply(self._A, X.contiguous())

    def P_mm(self, X: torch.Tensor) -> torch.Tensor:
        return _SymSpMM.apply(self._P, X.contiguous())

    def L_mm(self, X: torch.Tensor) -> torch.Tensor:
        return X - self._P @ X

    def RW_mm(self, X: torch.Tensor) -> torch.Tensor:
        return self._RW @ X

    def beta_bank(self, X: torch.Tensor, C: int) -> torch.Tensor:
        """(C+1, n, c) tensor whose slice nu is beta_nu(I - P) X."""
        powers = [X]
        for _ in range(C):
            powers.append(self.L_mm(powers[-1]))
        coef = torch.as_tensor(beta_coefficients(C), dtype=X.dtype, device=X.device)
        return torch.einsum("vk,knc->vnc", coef, torch.stack(powers))
