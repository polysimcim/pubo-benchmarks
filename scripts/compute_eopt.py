"""Наилучшая известная энергия для каждого тензора.

Минимум по финальным снимкам всех алгоритмов из solution_data (5000 реплик),
то же, что min_e_list в drow.ipynb. Свёртка считается чанками: матрица
внешних произведений на 5000 реплик сразу не помещается в память.
"""
import numpy as np, torch, json, os, time
torch.set_num_threads(4)
M = int(os.environ.get('M', 12))
ALGS = ['greedy', 'polySBM', 'polyVelox', 'polysimCIM', 'QUBO_SBM', 'Velox_q']


def energy_chunked(s, Tm, N, chunk=400):
    out = []
    for a in range(0, s.shape[0], chunk):
        x = s[a:a+chunk].float()
        outer = (x[:, :, None] * x[:, None, :]).reshape(x.shape[0], -1)
        field = outer @ Tm.T                      # [r, N]
        out.append((field * x).sum(1) / 6.0)
    return torch.cat(out)


res = {}
for j in range(M):
    t0 = time.time()
    T = torch.tensor(np.load(f'J_T_h/T_{j}.npy'), dtype=torch.float32)
    N = T.shape[0];  Tm = T.reshape(N, -1).contiguous()
    best, who = float('inf'), None
    for alg in ALGS:
        p = f'solution_data/{alg}_{j}.npy'
        if not os.path.exists(p):
            continue
        data = np.load(p, allow_pickle=True)
        e = float(energy_chunked(torch.as_tensor(np.asarray(data[-1])), Tm, N).min())
        if e < best:
            best, who = e, alg
        del data
    res[j] = best
    print(f'тензор {j:>2}: E_opt = {best:>14.1f}  ({who}),  {time.time()-t0:.0f} с', flush=True)

json.dump(res, open('e_opt.json', 'w'), indent=1)
print('-> e_opt.json')
