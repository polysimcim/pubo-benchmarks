"""eta в зависимости от величины отклонения гиперпараметров.

eta = E / E_opt, как в drow.ipynb (E_opt — наилучшая энергия для тензора;
на малой размерности берётся точный оптимум перебором). eta <= 1, больше лучше.

Для каждой величины отклонения d каждый гиперпараметр умножается на
(1 ± d) со случайным знаком; знаки разыгрываются заново для каждого тензора,
так что разброс по тензорам отражает и разные углы гиперкуба возмущений.
"""
import json, os, sys, time, itertools, zlib
import numpy as np, torch
import sens_lib

torch.set_num_threads(int(os.environ.get('THREADS', 1)))

N        = int(os.environ.get('N', 10))          # размерность тензора
M        = int(os.environ.get('M', 12))          # число тензоров
REPLICAS = int(os.environ.get('REPLICAS', 256))
DEVS     = [float(x) for x in os.environ.get('DEVS', '0,1,5,9,13,17').split(',')]
SHARD    = int(os.environ.get('SHARD', 0))
NSHARD   = int(os.environ.get('NSHARD', 1))
OUT      = os.environ.get('OUT', 'eta_sh0.npz')
TDIR     = os.environ.get('TDIR', 'J_T_h')   # каталог с тензорами T_*.npy

BASE = {
    'polySBM':    dict(dt=0.05, a0=0.75, threshold_scale=0.7, wall_position=1.0),
    'polyVelox':  dict(beta_start=0.01, beta_end=6.0, alpha_start=0.5, alpha_end=0.6),
    'polysimCIM': dict(sigma=0.001, dt=0.0004, zeta=1600.0, S=0.002, D=-0.3,
                       O=0.002, alpha=0.01),
}
STEPS = {'polySBM': 500, 'polyVelox': 1000, 'polysimCIM': 1000}

NS = sens_lib.load_solvers()
sys.path.insert(0, '.')
from pubo_to_qubo import PUBOToQUBOReducer


def gen_instances():
    """На рабочей размерности берём готовые тензоры, иначе генерируем."""
    if (os.path.exists(f'{TDIR}/T_0.npy')
            and np.load(f'{TDIR}/T_0.npy', mmap_mode='r').shape[0] == N):
        return [np.load(f'{TDIR}/T_{j}.npy').astype(np.float32) for j in range(M)]
    cells = [''.join(c['source'])
             for c in json.load(open('problem_generator.ipynb'))['cells']
             if c['cell_type'] == 'code']
    g = {}
    exec(compile(next(c for c in cells if 'class HardInstanceGenerator' in c),
                 'gen', 'exec'), g)
    out = []
    for k in range(M):
        gen = g['HardInstanceGenerator'](n_qubits=N, s2q=3, s3q=7,
                                         n_swap_layers=100, seed=1000 + k)
        _, T, _ = gen.generate()
        out.append(T.astype(np.float32))
    return out


def optimum(T, j):
    """E_opt: наилучшая известная энергия для этой размерности, иначе перебор.

    Таблица привязана к N: ключи в e_opt*.json — номера тензоров, и файл для
    N=200 нельзя применять к другому набору. Если таблицы нет и перебор
    невозможен, вернётся nan, а eta посчитается при построении графика как
    отношение к лучшему результату по всем прогонам на этом тензоре.
    """
    for f in (f'e_opt_{N}.json', 'e_opt.json' if N == 200 else None):
        if f and os.path.exists(f):
            tab = json.load(open(f))
            if str(j) in tab:
                return float(tab[str(j)])
    if N > 24:
        return float('nan')
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=N)), dtype=np.float32)
    e = sens_lib.energy(torch.tensor(S), torch.tensor(T))
    return float(e.min())


def perturb(alg, d, j):
    """Гиперпараметры, сдвинутые случайным вектором фиксированной длины.

    Вектор относительных изменений delta берётся равномерно по направлению,
    его длина фиксирована: ||delta|| = d*sqrt(k). При таком радиусе
    среднеквадратичное изменение одного параметра равно d, то есть ось графика
    сохраняет прежний смысл, но наборы становятся непрерывными, а не 2^k
    вершинами гиперкуба.

    Отдельная компонента может превысить единицу и увести параметр за ноль
    (sigma, dt, zeta должны оставаться положительными, D — отрицательным),
    поэтому такие направления отбраковываются.
    """
    base = BASE[alg]
    keys = sorted(base)
    k = len(keys)
    if k == 0 or d == 0:
        return dict(base)
    # зерно детерминированное: hash() от строк солится от запуска к запуску
    rng = np.random.default_rng([zlib.crc32(alg.encode()), int(round(d * 100)), j])
    r = d / 100 * np.sqrt(k)
    for _ in range(1000):
        g = rng.standard_normal(k)
        delta = g / np.linalg.norm(g) * r
        if np.all(np.abs(delta) < 0.95):
            break
    else:                                   # вырожденный случай: ужимаем вектор
        delta = np.clip(delta, -0.95, 0.95)
    return {key: base[key] * (1 + float(delta[i])) for i, key in enumerate(keys)}


def run(alg, T, hp, seed):
    torch.manual_seed(seed);  np.random.seed(seed)
    Z1 = np.zeros(N, dtype=np.float32);  Z2 = np.zeros((N, N), dtype=np.float32)
    if alg == 'polySBM':
        sol = NS['PolynomialSimulatedBifurcationMachine'](J=-Z2, T=-T, h=-Z1, device='cpu')
        cfg = NS['PolySBMConfig'](num_steps=STEPS[alg], num_replicas=REPLICAS,
                                  use_ternary=True, device='cpu', **hp)
        spins, _ = sol.solve(cfg)
    elif alg == 'polyVelox':
        sd = np.std(T)
        cfg = NS['polyVeloxConfig'](num_sweeps=STEPS[alg], num_samples=REPLICAS,
                                    annealing_schedule='exp', device='cpu',
                                    verbose=False, **hp)
        _, data = NS['polyVeloxSolver'](Z1/sd, Z2/sd, T/sd, cfg).solve()
        spins = data[-1]
    elif alg == 'polysimCIM':
        sim = NS['SIMCIM'](T, hp['O'], hp['S'], hp['D'], STEPS[alg], hp['dt'], N, N,
                           hp['sigma'], hp['zeta'], hp['alpha'], REPLICAS, device='cpu')
        _, data = sim.evolve()
        spins = data[-1]
    elif alg in ('QUBO_SBM', 'Velox_q'):
        J, b, _ = PUBOToQUBOReducer().reduce(Z1, Z2, T, p0=1)
        if alg == 'QUBO_SBM':
            cfg = NS['SBMConfig'](num_steps=STEPS[alg], num_replicas=REPLICAS,
                                  use_ternary=True, device='cpu', **hp)
            sol = NS['SimulatedBifurcationMachine'](-J/np.std(J), -b/np.std(J),
                                                    device='cpu')
            _, data = sol.solve(cfg)
            spins = data[-1]
        else:
            Q = NS['ising_to_qubo'](J, b)
            cfg = NS['VeloxQConfig'](num_sweeps=STEPS[alg], num_samples=REPLICAS,
                                     auto_tune=False, verbose=False, device='cpu',
                                     annealing_schedule='exp', **hp)
            _, data = NS['VeloxQSolver'](Q, config=cfg).solve()
            spins = data[-1]        # снимки VeloxQ уже в виде ±1 (в отличие от best_x)
    s = torch.as_tensor(np.asarray(spins), dtype=torch.float32)
    if s.ndim == 1:
        s = s[None]
    s = s[:, :N]                       # у QUBO-солверов обрезаем вспомогательные
    e = sens_lib.energy(s, torch.tensor(T))
    # сохраняем всё распределение по репликам: метрику можно будет сменить
    # (лучший / медианный / средний прогон) без повторного счёта
    return dict(min=float(e.min()), p25=float(e.quantile(0.25)),
                median=float(e.median()), mean=float(e.mean()),
                p75=float(e.quantile(0.75)), max=float(e.max()),
                n_replicas=int(e.numel()))


def main():
    Ts = gen_instances()
    opt = [optimum(T, j) for j, T in enumerate(Ts)]
    print('E_opt:', ' '.join(f'{o:.1f}' for o in opt), flush=True)

    jobs = [(alg, d, j) for alg in BASE for d in DEVS for j in range(M)]
    mine = [x for n, x in enumerate(jobs) if n % NSHARD == SHARD]

    # докатка: уже посчитанные задания из прошлого запуска не повторяем
    rows = []
    if os.path.exists(OUT):
        rows = [tuple(r) for r in np.load(OUT, allow_pickle=True)['rows']]
        done_keys = {(r[0], float(r[1]), int(r[2])) for r in rows}
        before = len(mine)
        mine = [x for x in mine if (x[0], float(x[1]), int(x[2])) not in done_keys]
        print(f'докатка: уже есть {len(rows)}, осталось {len(mine)} из {before}',
              flush=True)

    t0 = time.time()
    for n, (alg, d, j) in enumerate(mine):
        hp = perturb(alg, d, j)
        try:
            st = run(alg, Ts[j], hp, seed=5000 + j)
            eta = st['median'] / opt[j]
        except Exception as exc:
            print(f'  ОШИБКА {alg} d={d} t{j}: {type(exc).__name__}: {exc}', flush=True)
            st = dict(min=float('nan'), median=float('nan'), mean=float('nan'))
            eta = float('nan')
        rows.append((alg, d, j, st['min'], opt[j], st['min'] / opt[j],
                     json.dumps(hp), json.dumps(st)))
        if n % 5 == 0 or n == len(mine) - 1:
            el = time.time() - t0
            print(f'[{n+1:>4}/{len(mine)}] {alg:<11} d={d:>4.0f}% t{j:<3} eta={eta:.4f}  '
                  f'{el/60:.1f} мин, осталось ~{el/(n+1)*(len(mine)-n-1)/60:.0f} мин',
                  flush=True)
            np.savez(OUT, rows=np.array(rows, dtype=object),
                     meta=json.dumps(dict(N=N, M=M, REPLICAS=REPLICAS, DEVS=DEVS)))
    np.savez(OUT, rows=np.array(rows, dtype=object),
             meta=json.dumps(dict(N=N, M=M, REPLICAS=REPLICAS, DEVS=DEVS)))
    print('готово, строк:', len(rows))


if __name__ == '__main__':
    main()
