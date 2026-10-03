"""Searches for the additional tasks (XOR, PAT, NARMA): S0, S1, S4; mean field 5 seeds; 500 and 50000 receptors 3 seeds.
python launch_additional_tasks.py <n_parallel>
The searches (search_groups.py, group bits) write into data/results/additional_tasks_and_readouts/trials."""
import subprocess, sys, itertools, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
OUT = ROOT / "data" / "results" / "additional_tasks_and_readouts"; (OUT / "logs").mkdir(parents=True, exist_ok=True)
n_par = int(sys.argv[1]) if len(sys.argv) > 1 else 11
py = sys.executable; BO = HERE / "search_groups.py"; cmds = []
def cmd(s, t, mode, NR, seed):
    tag = f"bits_{mode}" + (f"_NR{NR}" if mode == "hyb" else "")
    return f'"{py}" "{BO}" --group bits --set {s} --tasks {t} --mode {mode} --NR {NR} --R 10 --seed {seed} --n-calls 100 --outdir "{OUT/"trials"}" > "{OUT/"logs"}/{tag}_{s}_{t}_seed{seed}.log" 2>&1'
for s, t, seed in itertools.product(("S0", "S1", "S4"), ("XOR", "PAT", "NARMA"), range(1, 6)):
    if s == "S0" and seed > 1: continue
    cmds.append(cmd(s, t, "det", 0, seed))
for NR in (500, 50000):
    for s, t, seed in itertools.product(("S0", "S1", "S4"), ("XOR", "PAT", "NARMA"), range(1, 4)):
        cmds.append(cmd(s, t, "hyb", NR, seed))
print(len(cmds), "commands", flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
with ThreadPoolExecutor(max_workers=n_par) as ex:
    codes = list(ex.map(lambda c: subprocess.run(c, shell=True, env=env, check=False).returncode, cmds))
print("searches finished, nonzero exit codes:", sum(1 for c in codes if c), flush=True)
