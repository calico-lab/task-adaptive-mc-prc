"""Readout grid for the mean-field optima (readout designed for noise). For every (task, set) optimum and receptor number,
all variants of readout_design.evaluate_variants (node count, low-pass time constant, readout memory) on R realizations.
Only the mean-field S1 and S4 optima of the benchmark tasks and of the groups case and bits are used.
Usage: python readout_grid.py <out_csv> <R> <n_workers> <NR list> <optima_csv> [optima_csv ...] [--tasks A,B] [--det]
Rows are appended to <out_csv>. With --det, the grid is evaluated in the mean field (N_R = 0, one realization).
The manuscript used R = 20, the NR list 500,5000,50000,500000, and the optima.csv files of the three folders in
data/results; its grids are data/results/additional_tasks_and_readouts/readout_grid.csv (receptor-count noise) and
readout_grid_mean_field.csv (--det).
"""
import csv, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import readout_design as m
mc = m.mc
args = sys.argv[1:]
tasks_filter = None; DET = False
if "--det" in args:
    DET = True; args = [x for x in args if x != "--det"]
if "--tasks" in args:
    i = args.index("--tasks"); tasks_filter = args[i + 1].split(","); args = args[:i] + args[i + 2:]
OUT = Path(args[0]); R = int(args[1]); NW = int(args[2]); NRS = [int(x) for x in args[3].split(",")]; FILES = args[4:]


def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def job(a):
    o, NR, k = a
    p = mc.params_from_row(o)
    rows = m.evaluate_variants(o["task"], p, 0 if DET else NR, 1 if DET else R, 70000 + 5 * k, mode="det" if DET else "hyb")
    for r in rows: r.update(set=o["set"], group=o.get("group", "benchmark"))
    return rows


if __name__ == "__main__":
    optima = [r for f in FILES for r in rd(f) if r["mode"] == "det" and r["set"] in ("S1", "S4") and r.get("group", "benchmark") in ("benchmark", "case", "bits") and (tasks_filter is None or r["task"] in tasks_filter)]
    jobs = [(o, NR, k) for k, (o, NR) in enumerate((o, NR) for o in optima for NR in ([0] if DET else NRS))]
    print(len(optima), "optima,", len(jobs), "jobs", flush=True)
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, rr in enumerate(ex.map(job, jobs)):
            write_header = not OUT.exists()
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
                if write_header: w.writeheader()
                w.writerows(rr)
            n += len(rr)
            best = min(rr, key=lambda r: r["objective"]); base = [r for r in rr if r["variant"] == "sampled_default"][0]
            print(f"{i+1}/{len(jobs)} {rr[0]['set']} {rr[0]['task']:8s} NR={rr[0]['N_R']:>7d} default {base['objective']:.3f} -> best {best['objective']:.3f} (Mp {best['Mp']}, tau/T {best['tau_over_T']}, L {best['L']}, lam {best['lam']:.0e}) {time.time()-t0:.0f} s", flush=True)
    print("appended", n, "rows", flush=True)
