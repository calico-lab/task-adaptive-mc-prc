"""Joint channel-plus-readout searches: S1 and S3, tasks MG, SINE, MGCUBED, NARMA, at 5000 and 50000 receptors, 3 seeds.
python launch_joint_readout.py <n_parallel>
The searches (search_joint_readout.py) write into data/results/additional_tasks_and_readouts/trials."""
import subprocess, sys, itertools, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
OUT = ROOT / "data" / "results" / "additional_tasks_and_readouts"; (OUT / "logs").mkdir(parents=True, exist_ok=True)
n_par = int(sys.argv[1]) if len(sys.argv) > 1 else 11; py = sys.executable; cmds = []
for NR, s, t, seed in itertools.product((5000, 50000), ("S1", "S3"), ("MG", "SINE", "MGCUBED", "NARMA"), range(1, 4)):
    cmds.append(f'"{py}" "{HERE/"search_joint_readout.py"}" --set {s} --task {t} --NR {NR} --R 10 --seed {seed} --n-calls 100 --outdir "{OUT/"trials"}" > "{OUT/"logs"}/readout_NR{NR}_{s}_{t}_seed{seed}.log" 2>&1')
print(len(cmds), "commands", flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
with ThreadPoolExecutor(max_workers=n_par) as ex:
    codes = list(ex.map(lambda c: subprocess.run(c, shell=True, env=env, check=False).returncode, cmds))
print("searches finished, nonzero exit codes:", sum(1 for c in codes if c), flush=True)
