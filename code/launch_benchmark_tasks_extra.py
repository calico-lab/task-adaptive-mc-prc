"""Additional noise-aware searches on the benchmark tasks at one more receptor number (S0 to S4, three tasks).
python launch_benchmark_tasks_extra.py <n_parallel> <NR> <n_seeds>; the manuscript used NR = 500000 with 3 seeds."""
import subprocess, sys, itertools, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
OUT = ROOT / "data" / "results" / "benchmark_tasks"; (OUT / "logs").mkdir(parents=True, exist_ok=True)
n_par, NR, n_seeds = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
py = sys.executable; cmds = []
for s, t, seed in itertools.product(("S0", "S1", "S2", "S3", "S4"), ("MG", "SINE", "MGCUBED"), range(1, n_seeds + 1)):
    cmds.append(f'"{py}" "{HERE/"search_benchmark_tasks.py"}" --set {s} --task {t} --mode hyb --NR {NR} --R 10 --seed {seed} --n-calls 100 --outdir "{OUT/"trials"}" > "{OUT/"logs"}/hyb_NR{NR}_{s}_{t}_seed{seed}.log" 2>&1')
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
with ThreadPoolExecutor(max_workers=n_par) as ex:
    codes = list(ex.map(lambda c: subprocess.run(c, shell=True, env=env, check=False).returncode, cmds))
print(len(cmds), "commands finished, nonzero:", sum(1 for c in codes if c), flush=True)
