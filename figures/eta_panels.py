"""Двухпанельная фигура: eta от величины расстройки, медиана и минимум по репликам."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATTERN = os.environ.get('PATTERN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sensitivity', 'eta75_sh*.npz'))
OUT     = os.environ.get('OUTPNG', 'eta75_panels.png')

STYLE = {'polySBM':    ('#1f77b4', 'o', '-'),
         'polyVelox':  ('#d62728', 'o', '-'),
         'polysimCIM': ('#2ca02c', 'o', '-')}
ORDER = ['polySBM', 'polyVelox', 'polysimCIM']

rows, meta = [], None
for f in sorted(glob.glob(PATTERN)):
    z = np.load(f, allow_pickle=True)
    rows += list(z['rows']);  meta = json.loads(str(z['meta']))

# E_opt на тензор — лучшая энергия по всем прогонам этого тензора
opt = {}
for r in rows:
    j = int(r[2]);  opt[j] = min(opt.get(j, np.inf), float(r[3]))

def collect(metric):
    d = {}
    for r in rows:
        val = json.loads(r[7])[metric]
        d.setdefault(r[0], {}).setdefault(float(r[1]), []).append(val / opt[int(r[2])])
    return d

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
for ax, metric, tag, name in [(axes[0], 'median', 'a', 'median replica'),
                              (axes[1], 'min', 'b', 'best replica')]:
    d = collect(metric)
    for alg in ORDER:
        if alg not in d:
            continue
        devs = sorted(d[alg])
        m  = [np.mean(d[alg][x]) for x in devs]
        lo = [np.percentile(d[alg][x], 25) for x in devs]
        hi = [np.percentile(d[alg][x], 75) for x in devs]
        c, mk, ls = STYLE[alg]
        ax.fill_between(devs, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(devs, m, ls, color=c, marker=mk, lw=2, ms=5,
                label=alg, zorder=3)
    ax.set_xlabel('deviation from tuned configuration, %')
    ax.set_title(f'({tag}) {name}', fontsize=11, pad=8)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel(r'$\eta = E / E_{\mathrm{opt}}$', fontsize=11)
axes[0].legend(fontsize=9.5, frameon=False, loc='lower left')
fig.suptitle(f'N={meta["N"]}, {meta["M"]} instances, {meta["REPLICAS"]} replicas; '
             f'lines: mean over instances, bands: interquartile range',
             fontsize=9.5, y=0.015, color='#444444')
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT, dpi=300, bbox_inches='tight')
pdf = os.path.splitext(OUT)[0] + '.pdf'
plt.savefig(pdf, bbox_inches='tight')      # вектор для вёрстки
print('->', OUT, 'и', pdf)
