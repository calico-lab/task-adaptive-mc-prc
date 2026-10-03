"""Input and target series of the additional tasks: temporal XOR and three-bit pattern detection on an on-off keyed bit
stream, and NARMA-10.

XOR    input b(n) i.i.d. Bernoulli(1/2), target y(n) = b(n) xor b(n-1)
PAT    same input, target y(n) = 1 if (b(n-2), b(n-1), b(n)) = (1, 0, 1) else 0
NARMA  input u(n) i.i.d. uniform on [0, 0.5], target y(n+1) = 0.3 y(n) + 0.05 y(n) sum_{i=0}^{9} y(n-i) + 1.5 u(n-9) u(n) + 0.1,
       the target at symbol n is y(n) (Atiya and Parlos 2000, as used throughout the reservoir-computing literature)
All series have 5000 symbols and depend on one seed. Output: data/task_inputs/additional_tasks.npz
Usage: python additional_task_series.py [seed]   (default seed 2029, the series used in the manuscript)
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "data" / "task_inputs" / "additional_tasks.npz"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2029; N = 5000
rng = np.random.default_rng(SEED)
b = rng.integers(0, 2, N).astype(float)
xor = np.zeros(N); xor[1:] = np.logical_xor(b[1:], b[:-1]).astype(float)
pat = np.zeros(N)
for n in range(2, N):
    pat[n] = 1.0 if (b[n - 2], b[n - 1], b[n]) == (1.0, 0.0, 1.0) else 0.0
u = rng.uniform(0.0, 0.5, N)
y = np.zeros(N)
for n in range(N - 1):
    lo = max(0, n - 9)
    y[n + 1] = 0.3 * y[n] + 0.05 * y[n] * y[lo:n + 1].sum() + 1.5 * (u[n - 9] if n >= 9 else 0.0) * u[n] + 0.1
np.savez(OUT, bits=b, target_xor=xor, target_pat=pat, narma_input=u, narma_target=y, seed=SEED)
print(f"saved {OUT}: bits mean {b.mean():.3f}, xor positive {xor.mean():.3f}, pattern positive {pat.mean():.3f}, narma target {y.min():.3f} to {y.max():.3f} (mean {y.mean():.3f})")
