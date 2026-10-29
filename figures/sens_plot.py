"""Разбор и график чувствительности к гиперпараметрам."""
import json, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = {
    'polySBM':    dict(dt=0.05, a0=0.75, threshold_scale=0.7, wall_position=1.0),
    'polyVelox':  dict(beta_start=0.01, beta_end=6.0, alpha_start=0.5, alpha_end=0.6),
    'polysimCIM': dict(sigma=0.001, dt=0.0004, zeta=1600.0, S=0.002, D=-0.3,
                       O=0.002, alpha=0.01),
    'greedy':     dict(),
}
TITLE = {'polySBM': 'polySBM', 'polyVelox': 'polyVelox',
         'polysimCIM': 'polysimCIM', 'greedy': 'greedy (no parameters)'}
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'best'   # best | mean


def load():
    rows = []
    for f in sorted(glob.glob('sens_sh*.npz')):
        rows += list(np.load(f, allow_pickle=True)['rows'])
    return rows


def collect(rows):
    """-> alg -> kind -> {seed: {inst: значение}} и факторы возмущений."""
    col = 4 if METRIC == 'best' else 5
    d, facts = {}, {}
    for alg, kind, seed, inst, eb, em, hp in rows:
        v = float(eb if METRIC == 'best' else em)
        d.setdefault(alg, {}).setdefault(kind, {}).setdefault(int(seed), {})[int(inst)] = v
        if kind == 'pert':
            hp = json.loads(hp)
            facts.setdefault(alg, {})[int(seed)] = {k: hp[k] / BASE[alg][k] for k in hp}
    return d, facts


def relative(d):
    """Относительное отклонение от базовой медианы, отдельно по каждому инстансу.

    Коэффициенты задач берутся из распределения Коши, поэтому абсолютные
    энергии у разных инстансов различаются в разы — усреднять их напрямую
    нельзя.
    """
    out = {}
    for alg, kinds in d.items():
        insts = sorted({i for k in kinds.values() for s in k.values() for i in s})
        ref = {}
        for i in insts:
            vals = [s[i] for s in kinds.get('base', {}).values() if i in s]
            ref[i] = np.median(vals) if vals else np.nan
        res = {}
        for kind, seeds in kinds.items():
            arr = []
            for seed, per_inst in seeds.items():
                r = [(per_inst[i] - ref[i]) / abs(ref[i])
                     for i in per_inst if np.isfinite(ref[i]) and ref[i] != 0]
                if r:
                    arr.append((seed, float(np.mean(r))))
            res[kind] = arr
        out[alg] = res
    return out


def main():
    rows = load()
    if not rows:
        print('нет данных'); return
    d, facts = collect(rows)
    rel = relative(d)
    algs = [a for a in BASE if a in rel]

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4),
                           gridspec_kw={'width_ratios': [1.15, 1]})

    # (а) разброс результата: возмущения на фоне шумового пола
    pos, labels = [], []
    for k, alg in enumerate(algs):
        for j, (kind, color, lab) in enumerate(
                [('base', '#8a8a8a', 'baseline configuration'),
                 ('pert', '#c0392b', 'perturbed by ±10%')]):
            v = [x for _, x in rel[alg].get(kind, [])]
            if not v:
                continue
            p = k + (j - 0.5) * 0.3
            ax[0].scatter(np.random.normal(p, 0.035, len(v)), np.array(v) * 100,
                          s=16, color=color, alpha=0.55, zorder=3,
                          label=lab if k == 0 else None)
            ax[0].boxplot([np.array(v) * 100], positions=[p], widths=0.22,
                          showfliers=False, medianprops=dict(color='k', lw=1.6))
        pos.append(k); labels.append(TITLE[alg])
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].set_xticks(pos); ax[0].set_xticklabels(labels, fontsize=9.5)
    ax[0].set_ylabel('energy deviation from baseline, %')
    ax[0].set_title('(a) spread of the result\n(above zero — worse than baseline)',
                    fontsize=10.5)
    ax[0].grid(alpha=0.25, axis='y'); ax[0].legend(fontsize=9, loc='upper left')

    # (б) сводная чувствительность: избыток разброса над шумовым полом
    names, noise, total = [], [], []
    for alg in algs:
        b = np.array([x for _, x in rel[alg].get('base', [])]) * 100
        p = np.array([x for _, x in rel[alg].get('pert', [])]) * 100
        if len(b) < 2 or len(p) < 2:
            continue
        names.append(TITLE[alg]); noise.append(b.std(ddof=1)); total.append(p.std(ddof=1))
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, noise, 0.4, color='#8a8a8a', label='noise floor (different seeds)')
    ax[1].bar(x + 0.2, total, 0.4, color='#c0392b', label='perturbed by ±10%')
    for i, (n, t) in enumerate(zip(noise, total)):
        extra = np.sqrt(max(t**2 - n**2, 0.0))
        ax[1].text(i + 0.2, t, f'parameter\ncontribution\n{extra:.2f}%', ha='center',
                   va='bottom', fontsize=8.2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=9.5)
    ax[1].set_ylabel('std. dev. of deviation, %')
    ax[1].set_title('(b) sensitivity to hyperparameters\n'
                    'spread in excess of intrinsic noise', fontsize=10.5)
    ax[1].grid(alpha=0.25, axis='y'); ax[1].legend(fontsize=9)
    ax[1].set_ylim(0, max(total + noise) * 1.35 if total else 1)

    plt.tight_layout()
    plt.savefig('sensitivity.png', dpi=130, bbox_inches='tight')
    print('-> sensitivity.png')

    # числовая сводка + вклад отдельных параметров (регрессия по факторам)
    print(f'\nmetric: E_{METRIC};  runs: {len(rows)}')
    print(f'{"algorithm":<14}{"n_base":>7}{"n_pert":>7}{"noise std%":>12}'
          f'{"perturbed std%":>16}{"contrib%":>10}')
    for alg in algs:
        b = np.array([x for _, x in rel[alg].get('base', [])]) * 100
        p = np.array([x for _, x in rel[alg].get('pert', [])]) * 100
        if len(b) < 2 or len(p) < 2:
            continue
        ex = np.sqrt(max(p.std(ddof=1)**2 - b.std(ddof=1)**2, 0.0))
        print(f'{alg:<14}{len(b):>7}{len(p):>7}{b.std(ddof=1):>10.3f}'
              f'{p.std(ddof=1):>14.3f}{ex:>9.3f}')

    print('\nper-parameter contribution (linear regression of deviation on factors):')
    for alg in algs:
        if not BASE[alg] or alg not in facts:
            continue
        pert = rel[alg].get('pert', [])
        keys = sorted(BASE[alg])
        X = np.array([[facts[alg][s][k] - 1.0 for k in keys] for s, _ in pert])
        y = np.array([v for _, v in pert]) * 100
        if len(y) < len(keys) + 3:
            continue
        A = np.hstack([np.ones((len(y), 1)), X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        eff = {k: coef[i + 1] * 0.1 for i, k in enumerate(keys)}   # эффект при +10%
        order = sorted(eff, key=lambda k: -abs(eff[k]))
        print(f'  {alg:<12} ' + ',  '.join(f'{k} {eff[k]:+.2f}%' for k in order))


if __name__ == '__main__':
    main()
