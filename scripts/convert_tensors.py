"""Перевод тензоров задач в сжатый формат .npz (float32).

Исходные .npy хранятся в float64, тогда как все решатели всё равно приводят
коэффициенты к float32, поэтому понижение точности не меняет результатов, а
размер набора сокращается с 5.1 ГБ до примерно 0.7 ГБ.
"""
import os, re, sys
import numpy as np

SRC = sys.argv[1]  # каталог с исходными .npy
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'tensors')

JOBS = [
    ('dense',         f'{SRC}/Dense',            r'K_dense_(\d+)_(\d+)_\.npy'),
    ('sparse_random', f'{SRC}/Sparse random',    r'K_sparse_random_(\d+)_(\d+)_\.npy'),
    ('sk_symmetric',  f'{SRC}/sk_symmetric',     r'^(\d+)_(\d+)\.npy$'),
    ('n200',          f'{SRC}/200/J_T_h',        r'^T_(\d+)\.npy$'),
    ('n200_alt',      f'{SRC}/200/J_T_h_1',      r'^T_(\d+)\.npy$'),
    ('n100',          f'{SRC}/200/J_T_h_100_v3', r'^T_(\d+)\.npy$'),
]

for name, src, pat in JOBS:
    out = os.path.join(DST, 'n200' if name.startswith('n') else name)
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if re.match(pat, f))
    done = raw = cmp_ = 0
    for f in files:
        m = re.match(pat, f)
        tag = '_'.join(m.groups())
        stem = f'{name}_{tag}' if name.startswith('n') else f'{tag}'
        dstf = os.path.join(out, stem + '.npz')
        T = np.load(os.path.join(src, f)).astype(np.float32)
        np.savez_compressed(dstf, T=T)
        raw += os.path.getsize(os.path.join(src, f))
        cmp_ += os.path.getsize(dstf)
        done += 1
    print(f'{name:<14} {done:>4} шт   {raw/2**20:>7.0f} МБ -> {cmp_/2**20:>6.1f} МБ', flush=True)
print('готово')
