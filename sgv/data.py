"""Graph loading.

One `.mat` file per graph, in the format used by common graph anomaly detection benchmarks:
    Network     n x n adjacency (dense or sparse; symmetrised, self-loops removed)
    Attributes  n x d node attributes (dense or sparse)
    Label       n anomaly labels, 1 = anomalous, 0 = normal (the key `gnd` is accepted too)
    keep_mask   optional: indices (or a boolean mask) of the labelled nodes; the graph is then restricted to the
                subgraph they induce, and only those nodes are scored
"""
from __future__ import annotations

import dataclasses as dc
import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


@dc.dataclass
class Graph:
    name: str
    A: sp.csr_matrix      # loop-free, symmetric, 0/1, float64
    y: np.ndarray         # int64, 1 = anomaly
    X: np.ndarray         # float64 attributes, n x d

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> np.ndarray:
        """Loop-free degree of every node."""
        return np.asarray(self.A.sum(1)).ravel()


def loop_free(A: sp.spmatrix) -> sp.csr_matrix:
    """Symmetric 0/1 CSR adjacency, self-loops and duplicates removed."""
    A = sp.csr_matrix(A)
    A = (A + A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    A.sort_indices()
    return A.astype(np.float64)


def load_graph(root: str, name: str) -> Graph:
    m = sio.loadmat(os.path.join(root, f"{name}.mat"))
    A = sp.csr_matrix(m["Network"])
    X = m["Attributes"]
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    y = np.squeeze(np.asarray(m["Label"] if "Label" in m else m["gnd"])).astype(np.int64)
    if "keep_mask" in m:
        k = np.squeeze(np.asarray(m["keep_mask"])).astype(np.int64)
        if k.size != X.shape[0] or not np.array_equal(np.sort(k), np.arange(X.shape[0])):
            idx = k if k.max() >= 2 else np.flatnonzero(k.astype(bool))    # index list vs boolean mask
            A, X, y = A[idx][:, idx], X[idx], y[idx]
    X = (X.toarray() if sp.issparse(X) else X).astype(np.float64)
    return Graph(name, loop_free(A), y, X)
