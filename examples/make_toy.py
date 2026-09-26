"""Write small synthetic graphs in the expected .mat format, to check the pipeline end to end without any dataset.

    python examples/make_toy.py --out toy_data
    python -m sgv.train --data_root toy_data --sources toy_a,toy_b --targets toy_c --set train.epochs=5

Each graph is a stochastic block model with Gaussian attributes; 5% of the nodes are made anomalous by
replacing their attributes with those of a random distant node and wiring them into a small clique.
The numbers it produces mean nothing; it only exercises the code.
"""
import argparse
import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def toy_graph(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    k = 4
    block = rng.integers(0, k, n)
    centers = rng.normal(0, 1, (k, d))
    X = centers[block] + 0.5 * rng.normal(0, 1, (n, d))
    p_in, p_out = 8.0 / n * k, 1.0 / n
    rows, cols = [], []
    for i in range(n):
        p = np.where(block == block[i], p_in, p_out)
        j = np.flatnonzero(rng.random(n) < p)
        rows += [i] * len(j)
        cols += list(j)
    y = np.zeros(n, np.int64)
    anom = rng.choice(n, n // 20, replace=False)
    y[anom] = 1
    X[anom] = X[rng.choice(n, anom.size)] + rng.normal(0, 3, (anom.size, d))
    for c in np.array_split(anom, max(1, anom.size // 5)):
        for a in c:
            rows += [a] * len(c)
            cols += list(c)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)
    return A, X, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy_data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, (n, d, seed) in {"toy_a": (800, 100, 0), "toy_b": (600, 30, 1), "toy_c": (700, 120, 2)}.items():
        A, X, y = toy_graph(n, d, seed)
        sio.savemat(os.path.join(a.out, f"{name}.mat"), {"Network": A, "Attributes": X, "Label": y[:, None]})
        print(f"wrote {name}.mat  n={n} d={d} anomalies={int(y.sum())}")
