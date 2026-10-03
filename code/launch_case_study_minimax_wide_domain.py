"""Command list of the searches for the case study, the wide domain, and the minimax (compromise) objective, run with a
fixed number of parallel processes. python launch_case_study_minimax_wide_domain.py <n_parallel>
The searches write into data/results/case_study_minimax_wide_domain/trials and their logs into its logs folder."""
import subprocess, sys, itertools, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
OUT = ROOT / "data" / "results" / "case_study_minimax_wide_domain"; (OUT / "logs").mkdir(parents=True, exist_ok=True)
n_par = int(sys.argv[1]) if len(sys.argv) > 1 else 11
py = sys.executable; cmds = []
def cmd(group, s, tasks, mode, NR, seed):
    tag = f"{group}_{mode}" + (f"_NR{NR}" if mode == "hyb" else "")
    return (f'"{py}" "{HERE/"search_groups.py"}" --group {group} --set {s} --tasks {tasks} --mode {mode} --NR {NR} --R 10 --seed {seed} --n-calls 100 '
            f'--outdir "{OUT/"trials"}" > "{OUT/"logs"}/{tag}_{s}_{tasks.replace(",", "+")}_seed{seed}.log" 2>&1')
# case study: S0, S1, S4 on GLUF and GLUE; mean field 5 seeds, 50000 receptors 3 seeds, S1 also at 500 receptors
for s, t, seed in itertools.product(("S0", "S1", "S4"), ("GLUF", "GLUE"), range(1, 6)):
    if s == "S0" and seed > 1: continue
    cmds.append(cmd("case", s, t, "det", 0, seed))
for s, t, seed in itertools.product(("S0", "S1", "S4"), ("GLUF", "GLUE"), range(1, 4)):
    cmds.append(cmd("case", s, t, "hyb", 50000, seed))
for t, seed in itertools.product(("GLUF", "GLUE"), range(1, 4)):
    cmds.append(cmd("case", "S1", t, "hyb", 500, seed)); cmds.append(cmd("case", "S0", t, "hyb", 500, seed))
# wide domain: S1 and S4 on the benchmark tasks
for s, t, seed in itertools.product(("S1", "S4"), ("MG", "SINE", "MGCUBED"), range(1, 6)):
    cmds.append(cmd("wide", s, t, "det", 0, seed))
for s, t, seed in itertools.product(("S1", "S4"), ("MG", "SINE", "MGCUBED"), range(1, 4)):
    cmds.append(cmd("wide", s, t, "hyb", 50000, seed))
# compromise: S1 and S4 over the three benchmark tasks
for s, seed in itertools.product(("S1", "S4"), range(1, 6)):
    cmds.append(cmd("compromise", s, "MG,SINE,MGCUBED", "det", 0, seed))
for s, seed in itertools.product(("S1", "S4"), range(1, 4)):
    cmds.append(cmd("compromise", s, "MG,SINE,MGCUBED", "hyb", 50000, seed))
(OUT / "commands.txt").write_text("\n".join(cmds) + "\n"); print(len(cmds), "commands", flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
with ThreadPoolExecutor(max_workers=n_par) as ex:
    codes = list(ex.map(lambda c: subprocess.run(c, shell=True, env=env, check=False).returncode, cmds))
print("searches finished, nonzero exit codes:", sum(1 for c in codes if c), flush=True)
