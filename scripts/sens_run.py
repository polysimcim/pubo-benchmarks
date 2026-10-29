"""Чувствительность алгоритмов PUBO к гиперпараметрам.

Схема: для каждого алгоритма берутся базовые гиперпараметры из его ноутбука,
затем каждый непрерывный параметр независимо умножается на U(0.9, 1.1).
Чтобы отделить влияние параметров от собственного разброса стохастического
алгоритма, отдельно считается «шумовой пол» — повторы базовой конфигурации с
разными зёрнами. Бюджет (число шагов и реплик) при возмущении не трогается.

Результат прогона — финальное состояние: E_best (минимум по репликам) и
E_mean (среднее по репликам), целевая функция sum_{i<j<k} T_ijk s_i s_j s_k.
"""
import json, os, sys, time
import numpy as np, torch
import sens_lib

torch.set_num_threads(int(os.environ.get('THREADS', 1)))

N_INST   = int(os.environ.get('N_INST', 2))
N_BASE   = int(os.environ.get('N_BASE', 6))
N_DRAW   = int(os.environ.get('N_DRAW', 14))
SHARD    = int(os.environ.get('SHARD', 0))      # номер воркера
NSHARD   = int(os.environ.get('NSHARD', 1))     # всего воркеров
REPLICAS = int(os.environ.get('REPLICAS', 32))
OUT      = os.environ.get('OUT', 'sensitivity_results.npz')

# базовые значения — ровно те, с которыми считались сохранённые результаты
BASE = {
    'polySBM':    dict(dt=0.05, a0=0.75, threshold_scale=0.7, wall_position=1.0),
    'polyVelox':  dict(beta_start=0.01, beta_end=6.0, alpha_start=0.5, alpha_end=0.6),
    'polysimCIM': dict(sigma=0.001, dt=0.0004, zeta=1600.0, S=0.002, D=-0.3,
                       O=0.002, alpha=0.01),
    # GradientDescent не использует ни одного параметра динамики (шаг
    # детерминированный, s = sign(-grad)), поэтому возмущать нечего — он
    # включён как эталон собственного разброса.
    'greedy':     dict(),
}
STEPS = {'polySBM': 500, 'polyVelox': 1000, 'polysimCIM': 1000, 'greedy': 500}

NS = sens_lib.load_solvers()
Z1 = np.zeros(200, dtype=np.float32)
Z2 = np.zeros((200, 200), dtype=np.float32)


def run(alg, T, T_t, hp, seed):
    torch.manual_seed(seed);  np.random.seed(seed)
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
        sol = NS['polyVeloxSolver'](Z1/sd, Z2/sd, T/sd, cfg)
        _, data = sol.solve()
        spins = data[-1] if data else None
    elif alg == 'greedy':
        cfg = NS['GradConfig'](num_steps=STEPS[alg], num_samples=REPLICAS,
                               temperature=0.05, device='cpu')
        sol = NS['GradientDescent'](-T, -Z2, -Z1, cfg)
        _, data = sol.solve()
        spins = data[-1]
    elif alg == 'polysimCIM':
        sim = NS['SIMCIM'](T, hp['O'], hp['S'], hp['D'], STEPS[alg], hp['dt'], 200, 200,
                           hp['sigma'], hp['zeta'], hp['alpha'], REPLICAS, device='cpu')
        _, data = sim.evolve()
        spins = data[-1]                   # SIMCIM уже пишет снимки как [реплики, N]
    s = torch.as_tensor(np.asarray(spins), dtype=torch.float32)
    if s.ndim == 1:
        s = s[None]
    e = sens_lib.energy(s, T_t)
    return float(e.min()), float(e.mean())


def main():
    rng = np.random.default_rng(12345)
    T_list, Tt_list = [], []
    for i in range(N_INST):
        T = np.load(f'J_T_h/T_{i}.npy').astype(np.float32)
        T_list.append(T);  Tt_list.append(torch.tensor(T))

    # полный список заданий, одинаковый у всех воркеров
    jobs = []
    for alg, base in BASE.items():
        for k in range(N_BASE):
            jobs.append((alg, 'base', dict(base), 1000 + k))
        for k in range(N_DRAW):
            f = {p_: float(rng.uniform(0.9, 1.1)) for p_ in base}
            jobs.append((alg, 'pert', {p_: base[p_] * f[p_] for p_ in base}, 2000 + k))
    mine = [j for n, j in enumerate(jobs) if n % NSHARD == SHARD]

    rows, t_start = [], time.time()
    total = len(mine) * N_INST
    done = 0
    for alg, kind, hp, seed in mine:
        for i in range(N_INST):
            try:
                eb, em = run(alg, T_list[i], Tt_list[i], hp, seed + 97 * i)
            except Exception as exc:
                print(f'  ОШИБКА {alg} {kind} inst{i}: {type(exc).__name__}: {exc}', flush=True)
                eb = em = float('nan')
            rows.append((alg, kind, seed, i, eb, em, json.dumps(hp)))
            done += 1
        el = time.time() - t_start
        print(f'[{done:>4}/{total}] {alg:<11} {kind} seed={seed}  E_best={rows[-1][4]:.0f}  '
              f'прошло {el/60:.1f} мин, осталось ~{el/max(done,1)*(total-done)/60:.0f} мин', flush=True)
        np.savez(OUT, rows=np.array(rows, dtype=object),
                 meta=json.dumps(dict(N_INST=N_INST, N_BASE=N_BASE, N_DRAW=N_DRAW,
                                      REPLICAS=REPLICAS, STEPS=STEPS, SHARD=SHARD)))
    print('готово, строк:', len(rows))


if __name__ == '__main__':
    main()
