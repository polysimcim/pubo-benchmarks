"""Time-to-target TTS99 for eta >= 0.99 (Tables IV and V of the paper).

TTS99 = tau * ln(1 - 0.99) / ln(1 - p), where p is the fraction of runs reaching
eta >= 0.99 and tau = T_batch / R is the effective time per run of a batch of R
parallel runs. Medians over instances are reported. If no run of an instance
reaches the target, only p < 1/R is known and TTS99 is replaced by its lower bound;
medians affected by such bounds are printed with ">".
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = 0.99


def tts(hits, R, tau):
    """Return (TTS99, is_lower_bound) for `hits` successes out of R runs."""
    if hits == 0:
        return tau * np.log(1 - 0.99) / np.log(1 - 1 / R), True
    p = hits / R
    if p >= 1:
        return tau, False
    return tau * np.log(1 - 0.99) / np.log(1 - p), False


def fmt(values):
    """Median over instances; a lower bound if any bound lies at or below it."""
    m = np.median([t for t, _ in values])
    bound = any(b and t <= m for t, b in values)
    return (">" if bound else "") + f"{m:.2g}"


def table_n100():
    """N = 100, 2000 runs per instance, batch times of Table III."""
    runs = os.path.join(ROOT, "data", "benchmark_runs")
    t_batch = {"polysimcim": 2.7, "hopfield": 1.2, "leleu": 4.4, "tgd": 24.3, "greedy": 8522.8}
    print("N = 100, median TTS99 [s]")
    for family in ("sk", "dense", "sparse"):
        E = {a: [np.load(os.path.join(runs, family, a, f"100_{i}.npy")).astype(np.float64) for i in range(10)]
             for a in t_batch}
        best = [min(E[a][i].min() for a in t_batch) for i in range(10)]
        row = {a: fmt([tts(int(np.sum(E[a][i] / best[i] >= TARGET)), E[a][i].size, t_batch[a] / 2000)
                       for i in range(10)])
               for a in t_batch}
        print(f"  {family:7s}", row)


def table_n200():
    """Dense N = 200 instances, energies of all replicas at t = 15 s."""
    z = np.load(os.path.join(ROOT, "data", "n200_results", "dense_energies_15s.npz"))
    algs = [k[2:] for k in z.files if k.startswith("E_")]
    E = {a: z[f"E_{a}"].astype(np.float64) for a in algs}
    best = np.min([E[a].min(axis=1) for a in algs], axis=0)
    print("N = 200 (dense), median TTS99 [s]")
    for a in algs:
        R = E[a].shape[1]
        tau = float(z[f"t_{a}"][0]) / R
        t = [tts(int(np.sum(E[a][j] / best[j] >= TARGET)), R, tau) for j in range(E[a].shape[0])]
        print(f"  {a:11s} {fmt(t)}")


if __name__ == "__main__":
    table_n100()
    table_n200()
