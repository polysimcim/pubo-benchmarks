"""Загрузка солверов из ноутбуков + целевая функция для анализа чувствительности.

Код алгоритмов не дублируется: нужные ячейки исполняются из исходных .ipynb,
поэтому эксперимент всегда идёт на актуальной версии солверов.
"""
import json, sys
import numpy as np, torch

import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

NBDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'notebooks', 'n200')

NB = {
    'polySBM':    ('poly_sbm.ipynb', 'PolynomialSimulatedBifurcationMachine'),
    'polyVelox':  ('poly_velox.ipynb', 'polyVeloxSolver'),
    'greedy':     ('gradient_descent.ipynb', 'GradientDescent'),
    'polysimCIM': ('poly_simcim.ipynb', 'SIMCIM'),
    'QUBO_SBM':   ('qubo_sbm.ipynb', 'SimulatedBifurcationMachine'),
    'Velox_q':    ('velox_q.ipynb', 'VeloxQSolver'),
}

# вспомогательные функции, лежащие в отдельных ячейках
HELPERS = [('velox_q.ipynb', 'def ising_to_qubo')]


def load_solvers():
    """Возвращает словарь с классами солверов и их конфигами."""
    ns = {}
    for alg, (nb, cls) in NB.items():
        cells = [''.join(c['source'])
                 for c in json.load(open(os.path.join(NBDIR, nb)))['cells'] if c['cell_type'] == 'code']
        src = next(c for c in cells if f'class {cls}' in c)
        # polySBM.solve() сейчас заканчивается `return None` — мониторинг в нём
        # закомментирован. Возвращаем финальное состояние, сам алгоритм не трогаем.
        src = src.replace('return None#best_spins, data_spins',
                          'return torch.sign(q), None')
        src = '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('%'))
        exec(compile(src, nb, 'exec'), ns)
    for nb, marker in HELPERS:
        cells = [''.join(c['source'])
                 for c in json.load(open(os.path.join(NBDIR, nb)))['cells'] if c['cell_type'] == 'code']
        src = next(c for c in cells if marker in c)
        src = '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('%'))
        exec(compile(src, nb, 'exec'), ns)
    return ns


def energy(spins, T_t):
    """E(s) = sum_{i<j<k} T_ijk s_i s_j s_k, как в compute_energy_sparse при H1=H2=0.

    T симметризован по всем перестановкам и нулевой на совпадающих индексах,
    поэтому сумма по i<j<k равна полной свёртке, делённой на 6.
    """
    if spins.ndim == 1:
        spins = spins[None]
    s = spins.to(T_t.dtype)
    return torch.einsum('ijk,ri,rj,rk->r', T_t, s, s, s) / 6.0
