"""Write the command list of the benchmark-task searches and run it with a fixed number of parallel processes.
python launch_benchmark_tasks.py <n_parallel> [det|hyb|all]
Mean field: S0 to S4 on the three benchmark tasks, ten searches each (one for S0). Noise-aware: S0 to S4 at 500, 5000,
and 50000 receptors, five searches each (the searches at 500000 receptors are run by launch_benchmark_tasks_extra.py).
The searches write into data/results/benchmark_tasks/trials and their logs into data/results/benchmark_tasks/logs.
"""
import subprocess, sys, itertools, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data" / "results" / "benchmark_tasks"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
n_par = int(sys.argv[1]) if len(sys.argv) > 1 else 11
which = sys.argv[2] if len(sys.argv) > 2 else "all"
py = sys.executable
cmds = []
sets, tasks = ("S0", "S1", "S2", "S3", "S4"), ("MG", "SINE", "MGCUBED")
if which in ("det", "all"):
    for s, t, seed in itertools.product(sets, tasks, range(1, 11)):
        if s == "S0" and seed > 1:
            continue
        cmds.append(f'"{py}" "{HERE/"search_benchmark_tasks.py"}" --set {s} --task {t} --mode det --seed {seed} --n-calls 100 --outdir "{OUT/"trials"}" > "{OUT/"logs"}/det_{s}_{t}_seed{seed}.log" 2>&1')
if which in ("hyb", "all"):
    for NR in (500, 5000, 50000):
        for s, t, seed in itertools.product(sets, tasks, range(1, 6)):
            cmds.append(f'"{py}" "{HERE/"search_benchmark_tasks.py"}" --set {s} --task {t} --mode hyb --NR {NR} --R 10 --seed {seed} --n-calls 100 --outdir "{OUT/"trials"}" > "{OUT/"logs"}/hyb_NR{NR}_{s}_{t}_seed{seed}.log" 2>&1')
list_path = OUT / f"commands_{which}.txt"
list_path.write_text("\n".join(cmds) + "\n")
print(len(cmds), "commands ->", list_path, flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
from concurrent.futures import ThreadPoolExecutor
def run(cmd):
    return subprocess.run(cmd, shell=True, env=env, check=False).returncode
with ThreadPoolExecutor(max_workers=n_par) as ex:
    codes = list(ex.map(run, cmds))
print("searches finished, nonzero exit codes:", sum(1 for c in codes if c), flush=True)
