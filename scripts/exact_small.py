"""Exact ground states of the small benchmark instances (N = 10 and 20) by exhaustive search.

For every instance of the three N <= 100 families the energy of all 2^N spin
configurations is evaluated, and the exact minimum is compared with the
best-found reference energy, i.e. the minimum over all runs of all algorithms
stored in data/benchmark_runs.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TENSORS = os.path.join(ROOT, "data", "tensors")


def load_tensor(family, name):
    with np.load(os.path.join(TENSORS, family, name + ".npz")) as f:
        return f["T"]

FAMILIES = {"sk_symmetric": "sk", "dense": "dense", "sparse_random": "sparse"}
ALGORITHMS = ("polysimcim", "hopfield", "leleu", "tgd", "greedy")
RUNS = os.path.join(ROOT, "data", "benchmark_runs")


def exact_minimum(T, chunk=1 << 16):
    n = T.shape[0]
    i, j, k = np.nonzero(np.triu(np.ones((n, n)), 1)[:, :, None] * np.triu(np.ones((n, n)), 1)[None])
    keep = (i < j) & (j < k)
    i, j, k = i[keep], j[keep], k[keep]
    w = T[i, j, k]
    best = np.inf
    for start in range(0, 1 << n, chunk):
        idx = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)
        s = 1 - 2 * ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
        e = (s[:, i] * s[:, j] * s[:, k]).astype(np.float64) @ w
        best = min(best, e.min())
    return best


def best_found(family_dir, name):
    values = []
    for alg in ALGORITHMS:
        path = os.path.join(RUNS, family_dir, alg, f"{name}.npy")
        if os.path.exists(path):
            values.append(np.min(np.load(path)))
    return min(values)


if __name__ == "__main__":
    for family, family_dir in FAMILIES.items():
        for n in (10, 20):
            for idx in range(10):
                name = f"{n}_{idx}"
                T = load_tensor(family, name).astype(np.float64)
                e_exact, e_best = exact_minimum(T), best_found(family_dir, name)
                ok = np.isclose(e_exact, e_best, rtol=1e-5, atol=1e-6)
                print(f"{family:14s} {name:6s} exact={e_exact:14.6f} best-found={e_best:14.6f} {'OK' if ok else 'MISMATCH'}")
