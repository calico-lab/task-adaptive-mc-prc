"""Receptor-number sweep extended to engineered-sensor scales.
Re-evaluates mean-field optima (S0, S1, S4) with the readout retrained under noise (L and lambda re-selected) at
N_R = 1e6, 1e7, 1e8, 1e9 with R realizations. Input: one or more optima CSVs; the mean-field rows of the benchmark
tasks and of the groups 'case' and 'bits' are used.
Usage: python receptor_sweep_extended.py <out_csv> <R> <n_workers> <optima_csv> [optima_csv ...]
Rows are appended to <out_csv>; the sweep of the manuscript (R = 20, the optima.csv files of the three folders in
data/results) is data/results/additional_tasks_and_readouts/receptor_sweep_extended.csv.
"""
import csv, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import simulator as mc
OUT = Path(sys.argv[1]); R = int(sys.argv[2]); NW = int(sys.argv[3]); FILES = sys.argv[4:]
NRS = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]


def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def fl(x):
    try: return float(x)
    except Exception: return float("nan")


def job(args):
    o, NR, k = args
    p = mc.params_from_row(o); task = o["task"]; seed = 60000 + 3 * k
    b = mc.best_over_L(mc.evaluate_all_L(task, p, mode="hyb", N_R=NR, R=R, seed=seed))
    d = mc.evaluate(task, p, mode="det", lams=(fl(o["lam_best"]),))
    return dict(group=o.get("group", "benchmark"), set=o["set"], task=task, design_mode=o["mode"], eval_NR=NR, R=R, protocol="retrained",
                L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=b["opt_sd"], held=b["held"], meanfield=d["objective"])


if __name__ == "__main__":
    optima = []
    for f in FILES:
        for r in rd(f):
            if r["mode"] == "det" and r["set"] in ("S0", "S1", "S4") and r.get("group", "benchmark") in ("benchmark", "case", "bits"):
                optima.append(r)
    jobs = [(o, NR, k) for k, (o, NR) in enumerate((o, NR) for o in optima for NR in NRS)]
    print(len(optima), "optima,", len(jobs), "jobs", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r); print(f"{i+1}/{len(jobs)} {r['set']} {r['task']:8s} NR={r['eval_NR']:>10d} obj {r['objective']:.4f} (mean field {r['meanfield']:.4f}) L={r['L']} lam={r['lam']:.0e} {time.time()-t0:.0f} s", flush=True)
    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        w.writerows(rows)
    print("appended", len(rows), "rows to", OUT)
