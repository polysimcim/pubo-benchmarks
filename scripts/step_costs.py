"""Computational cost per step for N = 200 (Tables VI and VII of the paper).

Total time, total FLOPs (PyTorch profiler, `with_flops=True`) and number of steps
of one solve of instance 0 with R = 5000 runs on an NVIDIA RTX 2080 Ti, as printed
by the profiling cells of the N = 200 solver notebooks.
"""
R = 5000

# algorithm: (variables, steps, total FLOPs, total time [s])
MEASURED = {
    "dense": {
        "PolySimCIM": (200, 1000, 8.44e13, 25.073),
        "PolySBM": (200, 500, 4.04e13, 12.228),
        "PolyVelox": (200, 1000, 8.48e13, 26.143),
        "Greedy": (200, 500, 4.44e13, 16.413),
        "SBM": (19263, 400, 1.48e15, 212.785),
        "Velox": (19263, 500, 1.90e15, 174.044),
    },
    "sparse": {
        "PolySimCIM": (200, 1000, 8.44e13, 25.128),
        "PolySBM": (200, 500, 4.04e13, 12.344),
        "PolyVelox": (200, 1000, 8.48e13, 26.115),
        "Greedy": (200, 500, 4.44e13, 18.570),
        "SBM": (5453, 400, 1.19e14, 15.934),
        "Velox": (5453, 500, 1.49e14, 12.626),
    },
}

if __name__ == "__main__":
    for family, rows in MEASURED.items():
        print(f"{family}: variables, steps, FLOP/step, ms/step, updates/s, TFLOP/s, coupling memory [MB]")
        for alg, (n, steps, flops, t) in rows.items():
            mem = (200 ** 3 if n == 200 else n * n) * 4 / 1e6
            print(f"  {alg:10s} {n:6d} {steps:5d} {flops / steps:.1e} {1e3 * t / steps:6.0f} "
                  f"{R * steps / t:.1e} {flops / t / 1e12:5.1f} {mem:7.0f}")
