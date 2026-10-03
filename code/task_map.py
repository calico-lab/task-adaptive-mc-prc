"""Assemble the task map: for every task, the error of the mean-field S1 and S4 optima under noise against the receptor
number (default retrained readout, protocol 'retrained', and the selected readout of the readout grid, protocol
'remedy'), the noise-aware optima at their design receptor numbers, the linear input reference, and the critical
receptor numbers (receptor number at which the channel error falls below the linear reference, or below NRMSE 0.25,
log-interpolated; 'none' if never within the sweep). Reads the result files of the three folders in data/results.
Outputs in data/results/additional_tasks_and_readouts: task_map.csv (task, set, N_R, protocol, objective, held) and
critical_receptor_numbers.csv. Usage: python task_map.py
"""
import csv, math
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "data" / "results"
BENCH_DIR, GROUPS_DIR, ADD_DIR = RES / "benchmark_tasks", RES / "case_study_minimax_wide_domain", RES / "additional_tasks_and_readouts"
TASKS = ["MG", "SINE", "MGCUBED", "XOR", "PAT", "NARMA", "GLUF", "GLUE"]
CLASS = {"MG": "memory", "GLUF": "memory", "NARMA": "memory", "SINE": "nonlinearity", "XOR": "nonlinearity", "GLUE": "nonlinearity", "MGCUBED": "mixed", "PAT": "mixed"}


def rd(p):
    p = Path(p); return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if p.exists() else []
def fl(x):
    try: return float(x)
    except Exception: return float("nan")


rows = []
# linear references
ref = {}
for f in (BENCH_DIR / "baseline_taps.csv", GROUPS_DIR / "baseline_taps.csv", ADD_DIR / "baseline_taps.csv"):
    for r in rd(f): ref[r["task"]] = min(ref.get(r["task"], 9.0), fl(r["objective"]))
# mean-field values and retrained sweeps
for f, kind in ((BENCH_DIR / "reevaluation.csv", "benchmark"), (GROUPS_DIR / "reevaluation.csv", "case"), (ADD_DIR / "reevaluation.csv", "bits")):
    for r in rd(f):
        task = r.get("eval_task", r.get("task")); dtask = r.get("design_task", task)
        if task != dtask or r["set"] not in ("S0", "S1", "S4"): continue
        if kind == "case" and r.get("kind") != "case": continue
        nr = int(fl(r["eval_NR"]))
        if r["design_mode"] == "det" and r["protocol"] == "designed" and nr == 0 and r.get("node", "sample") == "sample":
            rows.append(dict(task=task, set=r["set"], N_R=0, protocol="meanfield", objective=fl(r["objective"]), held=fl(r["held"])))
        elif r["design_mode"] == "det" and r["protocol"] == "retrained" and nr > 0 and r.get("node", "sample") == "sample":
            rows.append(dict(task=task, set=r["set"], N_R=nr, protocol="retrained", objective=fl(r["objective"]), held=fl(r["held"])))
        elif r["design_mode"] == "hyb" and r["protocol"] == "designed" and nr == int(fl(r["design_NR"])):
            rows.append(dict(task=task, set=r["set"], N_R=nr, protocol="noiseaware", objective=fl(r["objective"]), held=fl(r["held"])))
for r in rd(ADD_DIR / "receptor_sweep_extended.csv"):
    rows.append(dict(task=r["task"], set=r["set"], N_R=int(fl(r["eval_NR"])), protocol="retrained", objective=fl(r["objective"]), held=fl(r["held"])))
# selected readout of the readout grid per (task, set, N_R), which includes the default readout of the same run
rem = defaultdict(list)
for f in (ADD_DIR / "readout_grid.csv", ADD_DIR / "readout_grid_mean_field.csv"):
    for r in rd(f):
        rem[(r["task"], r["set"], int(fl(r["N_R"])))].append(r)
for key, rr in rem.items():
    # the designed readout is the best of the integrating-node grid AND the sampled default evaluated on the same realizations
    best = min([r for r in rr if r["variant"] in ("integrating", "sampled_default")], key=lambda r: fl(r["objective"]))
    proto = "remedy_meanfield" if key[2] == 0 else "remedy"
    rows.append(dict(task=key[0], set=key[1], N_R=key[2], protocol=proto, objective=fl(best["objective"]), held=fl(best["held"]), Mp=best["Mp"], tau_over_T=best["tau_over_T"], L=best["L"], lam=best["lam"]))
# joint channel + readout searches (fresh values if available, else search objective)
fresh = rd(ADD_DIR / "joint_readout_reevaluation.csv")
for r in (fresh if fresh else rd(ADD_DIR / "joint_readout_optima.csv")):
    rows.append(dict(task=r["task"], set=r["set"], N_R=int(fl(r["N_R"])), protocol="joint", objective=fl(r["objective"]), held=fl(r["held"]), Mp=r["Mp"], tau_over_T=r["tau_over_T"], L=r["L"], lam=r["lam"]))
cols = ["task", "set", "N_R", "protocol", "objective", "held", "Mp", "tau_over_T", "L", "lam"]
with open(ADD_DIR / "task_map.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


LEVEL = 0.25   # absolute accuracy level for the second budget criterion


def budget(task, s, proto, level=None):
    pts = sorted([(r["N_R"], r["objective"]) for r in rows if r["task"] == task and r["set"] == s and r["protocol"] == proto and r["N_R"] > 0])
    if proto == "remedy" and pts:
        # the designed readout contains the default settings as one of its options; beyond the receptor numbers of the
        # readout grid (5e5) the retrained default values continue the curve
        nmax = pts[-1][0]
        pts += sorted([(r["N_R"], r["objective"]) for r in rows if r["task"] == task and r["set"] == s and r["protocol"] == "retrained" and r["N_R"] > nmax])
    L = ref.get(task) if level is None else level
    if not pts or L is None: return None
    if pts[0][1] < L: return pts[0][0]
    for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
        if e0 >= L > e1:
            x = math.log10(n0) + (e0 - L) / (e0 - e1) * (math.log10(n1) - math.log10(n0)); return 10 ** x
    return None


with open(ADD_DIR / "critical_receptor_numbers.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "class", "linear_ref", "mf_S1", "mf_S4", "budget_retrained_S1", "budget_retrained_S4", "budget_remedy_S1", "budget_remedy_S4", "level_retrained_S1", "level_retrained_S4", "level_remedy_S1", "level_remedy_S4"])
    for t in TASKS:
        mf = {s: [r["objective"] for r in rows if r["task"] == t and r["set"] == s and r["protocol"] == "meanfield"] for s in ("S1", "S4")}
        b = [budget(t, s, p) for s, p in (("S1", "retrained"), ("S4", "retrained"), ("S1", "remedy"), ("S4", "remedy"))]
        lv = [budget(t, s, p, LEVEL) for s, p in (("S1", "retrained"), ("S4", "retrained"), ("S1", "remedy"), ("S4", "remedy"))]
        w.writerow([t, CLASS[t], repr(ref.get(t, float('nan'))), repr(mf['S1'][0]) if mf["S1"] else "", repr(mf['S4'][0]) if mf["S4"] else ""] + [repr(x) if x else "none" for x in b] + [repr(x) if x else "none" for x in lv])
        print(f"{t:8s} {CLASS[t]:12s} ref {ref.get(t, float('nan')):.3f} MF S1 {mf['S1'][0] if mf['S1'] else float('nan'):.3f} S4 {mf['S4'][0] if mf['S4'] else float('nan'):.3f} | beat ref: ret S1 {b[0] and f'{b[0]:.2g}'} S4 {b[1] and f'{b[1]:.2g}'} rem S1 {b[2] and f'{b[2]:.2g}'} S4 {b[3] and f'{b[3]:.2g}'} | NRMSE<={LEVEL}: ret S1 {lv[0] and f'{lv[0]:.2g}'} S4 {lv[1] and f'{lv[1]:.2g}'} rem S1 {lv[2] and f'{lv[2]:.2g}'} S4 {lv[3] and f'{lv[3]:.2g}'}")
print(len(rows), "rows in task_map.csv")
