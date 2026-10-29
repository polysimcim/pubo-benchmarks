"""eta vs hyperparameter deviation.

Usage: python3.9 eta_plot.py [metric]
  metric: median (default) | min | mean | p25 | p75
  'min' reproduces the best-replica metric; 'median' is the typical run
  among the parallel replicas of one algorithm on one tensor.
"""
import json, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'median'
LABEL  = {'min': 'best replica', 'median': 'median replica', 'mean': 'mean replica',
          'p25': '25th pct replica', 'p75': '75th pct replica'}[METRIC]

ORDER  = ['polySBM', 'polyVelox', 'polysimCIM', 'QUBO_SBM', 'Velox_q']
COLORS = {'polySBM': '#1f77b4', 'polyVelox': '#d62728', 'polysimCIM': '#2ca02c',
          'QUBO_SBM': '#9467bd', 'Velox_q': '#ff7f0e'}

PATTERN = __import__('os').environ.get('PATTERN', 'eta_sh*.npz')
rows = []
for f in sorted(glob.glob(PATTERN)):
    z = np.load(f, allow_pickle=True)
    rows += list(z['rows']);  meta = json.loads(str(z['meta']))

# E_opt на тензор: берём из данных, иначе — лучший результат по всем прогонам
# этого тензора (все алгоритмы и все отклонения), как best-known в drow.ipynb
opt_tab, pooled = {}, {}
for r in rows:
    j, opt = int(r[2]), float(r[4])
    pooled[j] = min(pooled.get(j, np.inf), float(r[3]))
    if np.isfinite(opt):
        opt_tab[j] = opt
for j, v in pooled.items():
    opt_tab.setdefault(j, v)

d = {}
for r in rows:
    alg, dev, j = r[0], float(r[1]), int(r[2])
    if len(r) >= 8 and r[7]:
        val = json.loads(r[7])[METRIC]
    elif METRIC == 'min':
        val = float(r[3])
    else:
        continue                      # в старых данных есть только лучший прогон
    eta = val / opt_tab[j]
    if np.isfinite(eta):
        d.setdefault(alg, {}).setdefault(dev, []).append(float(eta))

if not d:
    sys.exit(f'нет данных для метрики {METRIC}')

fig, ax = plt.subplots(figsize=(9, 5.8))
for alg in ORDER:
    if alg not in d:
        continue
    devs = sorted(d[alg])
    m  = [np.mean(d[alg][x]) for x in devs]
    lo = [np.percentile(d[alg][x], 25) for x in devs]
    hi = [np.percentile(d[alg][x], 75) for x in devs]
    c = COLORS[alg]
    ax.fill_between(devs, lo, hi, color=c, alpha=0.18, lw=0)
    ax.plot(devs, m, '-o', color=c, lw=2, ms=5, label=alg)

ax.set_xlabel('hyperparameter deviation from baseline, %')
ax.set_ylabel(r'$\eta = E\,/\,E_{\mathrm{opt}}$')
ax.set_title(f'Sensitivity to hyperparameters ({LABEL})\n'
             f'N={meta["N"]}, {meta["M"]} tensors, {meta["REPLICAS"]} replicas; '
             f'line: mean over tensors, shading: 25th-75th percentile',
             fontsize=10.5)
ax.grid(alpha=0.25)
ax.legend(fontsize=9.5)
plt.tight_layout()
out = __import__('os').environ.get('OUTPNG', f'eta_vs_deviation_{METRIC}.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print('->', out)

devs_all = sorted(d[next(a for a in ORDER if a in d)])
print(f'\nmetric: {METRIC} ({LABEL})')
print(f'{"algorithm":<12}' + ''.join(f'{x:>9.0f}%' for x in devs_all))
for alg in ORDER:
    if alg in d:
        print(f'{alg:<12}' + ''.join(f'{np.mean(d[alg][x]):>10.4f}' for x in devs_all))
