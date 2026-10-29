# PUBO benchmarks

Code, instances and results for the paper *Polynomial unconstrained binary optimization inspired by optical simulation* (PolySimCIM).

## Structure

- `src/` — shared code: problem I/O and energy (`pubo.py`), PUBO→QUBO reduction (`pubo_to_qubo.py`), C++ PolySimCIM for quantum chemistry (`quantum_chemistry_cpp/`).
- `notebooks/` — solvers and experiments:
  - `sk_symmetric/`, `sparse_random/`, `cross_family/` — benchmarks for N ≤ 100 (Sec. III A);
  - `n200/` — N = 200 benchmarks with SBM, Velox and their polynomial versions (Sec. III C);
  - `quantum_chemistry/` — electronic structure (Sec. IV B).
- `scripts/` — time-to-target, Tables IV–V (`tts.py`), cost per step, Tables VI–VII (`step_costs.py`), exact ground states for N = 10, 20 (`exact_small.py`), sensitivity analysis, Appendix C (`sens_run.py`, `eta_run.py`), reference energies (`compute_eopt.py`), data utilities.
- `data/` — instances (`tensors/`), solver outputs for N ≤ 100 (`benchmark_runs/`) and N = 200 (`n200_results/`, energies of all runs at t = 15 s), sensitivity results (`sensitivity/`), quantum chemistry data (`quantum_chemistry/`).
- `figures/` — figures and the scripts that plot them.

## Paper ↔ repository

| Paper | Code | Data |
|---|---|---|
| Sec. II, Table I (PolySimCIM) | `notebooks/*/poly_simcim.ipynb` | — |
| Sec. III A, Fig. 1 (N ≤ 100) | `notebooks/sk_symmetric/`, `notebooks/sparse_random/`, `notebooks/cross_family/benchmark_figures.ipynb` | `data/tensors/{sk_symmetric,dense,sparse_random}`, `data/benchmark_runs/` |
| Table IV (TTS, N = 100) | `scripts/tts.py` | `data/benchmark_runs/` |
| Sec. III C, Fig. 4 (N = 200) | `notebooks/n200/` (instance generator, PolySimCIM, PolySBM, PolyVelox, SBM, Velox, Greedy, figures) | `data/tensors/n200/` |
| PUBO→QUBO reduction (Sec. III C) | `src/pubo_to_qubo.py` | — |
| Table V (TTS, N = 200) | `scripts/tts.py` | `data/n200_results/` |
| Tables VI–VII (cost per step, N = 200) | `scripts/step_costs.py` | — |
| Sec. IV B, Figs. 8–9 (electronic structure) | `notebooks/quantum_chemistry/`, `src/quantum_chemistry_cpp/` | `data/quantum_chemistry/` |
| Appendix B, Table X (tuned hyperparameters) | M-LOOP cells in the solver notebooks | `data/benchmark_runs/*/*/*_params*.npy` |
| Appendix C, Fig. 10 (sensitivity) | `scripts/eta_run.py`, `figures/eta_panels.py` | `data/sensitivity/` |

## Usage

Instances are stored as compressed `.npz`. To recreate the layout the notebooks expect, run:

```
python scripts/materialize_legacy_layout.py
```

The quantum chemistry notebooks require `openfermion`, `openfermionpsi4` and `psi4`. The C++ solver requires CMake, Boost and OpenMP.
