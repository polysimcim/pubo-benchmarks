"""Common utilities for the PUBO benchmark: problem I/O, energy, device choice.

The benchmark instances are cubic (3-local) Ising problems

    E(s) = sum_{i<j<k} T_ijk s_i s_j s_k ,    s_i = +-1 ,

where ``T`` is fully symmetric under permutation of its indices and vanishes
whenever two indices coincide. Four instance families are provided, differing
in how the non-zero triples are placed; see ``data/tensors`` and the README.
"""
from __future__ import annotations

import os
import re
from typing import Iterator, Tuple

import numpy as np

try:
    import torch
except ImportError:                      # torch is optional for pure-numpy use
    torch = None

DATA_ROOT = os.environ.get(
    "PUBO_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "tensors"),
)

FAMILIES = ("dense", "sparse_random", "sk_symmetric", "n200")


def device(prefer: str | None = None) -> "torch.device":
    """Pick a compute device.

    Honours ``$PUBO_DEVICE`` when set, then the explicit argument, then falls
    back to CUDA if available and finally to CPU. The published runs used a
    single GPU; nothing in the code requires a specific device index.
    """
    name = os.environ.get("PUBO_DEVICE") or prefer
    if name is None:
        name = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    return torch.device(name)


def load_tensor(family: str, name: str) -> np.ndarray:
    """Load one instance, e.g. ``load_tensor("dense", "100_0")``."""
    path = os.path.join(DATA_ROOT, family, f"{name}.npz")
    with np.load(path) as f:
        return f["T"]


def iter_tensors(family: str, n: int | None = None
                 ) -> Iterator[Tuple[str, np.ndarray]]:
    """Iterate over ``(name, T)`` of a family, optionally restricted to size ``n``."""
    folder = os.path.join(DATA_ROOT, family)
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".npz"):
            continue
        stem = fn[:-4]
        if n is not None:
            m = re.search(r"(\d+)_\d+$", stem)
            if m is None or int(m.group(1)) != n:
                continue
        yield stem, load_tensor(family, stem)


def energy(spins, T) -> "torch.Tensor":
    """Energy of one or more spin configurations.

    ``spins`` has shape ``(N,)`` or ``(batch, N)`` with entries +-1. Because
    ``T`` is fully symmetric and zero on repeated indices, the sum over
    ``i<j<k`` equals the full contraction divided by six.
    """
    if torch is None:
        raise ImportError("energy() requires PyTorch")
    T = torch.as_tensor(T)
    s = torch.as_tensor(spins, dtype=T.dtype, device=T.device)
    if s.ndim == 1:
        s = s[None]
    return torch.einsum("ijk,ri,rj,rk->r", T, s, s, s) / 6.0


def energy_numpy(spins: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NumPy counterpart of :func:`energy`, for environments without PyTorch."""
    s = np.atleast_2d(np.asarray(spins, dtype=T.dtype))
    return np.einsum("ijk,ri,rj,rk->r", T, s, s, s) / 6.0
