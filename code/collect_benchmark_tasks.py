"""Collect the benchmark-task searches into summary tables.

Writes into <results_dir> (default data/results/benchmark_tasks):
  searches.csv         one row per search (best evaluation, with parameters and descriptors)
  optima.csv           one row per (mode, N_R, set, task): the best search (lowest objective) plus
                       mean, sd, median, min, max of the per-search best objectives and held-out values
  running_min.csv      mean over searches of the running minimum against the evaluation index
  s0.csv               S0 enumeration (every L) per mode and N_R (mean over chain seeds for hyb)
Usage: python collect_benchmark_tasks.py [results_dir]
"""
import csv, glob, json, sys, statistics as st
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "results" / "benchmark_tasks"
TASKS = ("MG", "SINE", "MGCUBED"); SETS = ("S0", "S1", "S2", "S3", "S4")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


searches, running, s0rows = [], defaultdict(list), []
for done in sorted(glob.glob(str(RES / "trials" / "*" / "*.done"))):
    info = json.load(open(done))
    rows = rd(done.replace(".done", ".csv"))
    fin = [r for r in rows if r["is_finite"] == "1"]
    if not fin:
        continue
    key = (info["mode"], int(info["N_R"]), info["set"], info["task"])
    best = min(fin, key=lambda r: fl(r["objective"]))
    out = dict(mode=info["mode"], N_R=int(info["N_R"]), set=info["set"], task=info["task"], search_seed=int(info["search_seed"]),
               chain_seed=int(info["chain_seed"]), n_evaluations=len(rows), n_finite=len(fin), seconds=fl(info["seconds"]),
               best_eval_index=int(best["eval_index"]))
    for k, v in best.items():
        if k not in ("set", "search_seed", "eval_index", "is_finite", "mode", "N_R", "task", "running_min", "seed"):
            out[k] = v
    searches.append(out)
    if info["set"] != "S0":
        rm = np.array([fl(r["running_min"]) for r in sorted(rows, key=lambda r: int(r["eval_index"]))])
        running[key].append(rm)
    else:
        for r in rows:
            s0rows.append(dict(mode=info["mode"], N_R=int(info["N_R"]), task=info["task"], chain_seed=int(info["chain_seed"]), L=int(r["L"]),
                               objective=fl(r["objective"]), held=fl(r["held"]), lam_best=fl(r["lam_best"]), opt_sd=fl(r.get("opt_sd", "nan"))))

if not searches:
    sys.exit("no finished searches")
cols = list(searches[0].keys())
for s in searches:
    for k in s:
        if k not in cols: cols.append(k)
with open(RES / "searches.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(searches)

# optima
groups = defaultdict(list)
for s in searches:
    groups[(s["mode"], s["N_R"], s["set"], s["task"])].append(s)
optima = []
for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], TASKS.index(k[3]))):
    g = groups[key]
    obj = [fl(s["objective"]) for s in g]; held = [fl(s["held"]) for s in g]
    best = min(g, key=lambda s: fl(s["objective"]))
    row = dict(best)
    row.update(n_searches=len(g), obj_mean=st.mean(obj), obj_sd=st.stdev(obj) if len(obj) > 1 else 0.0, obj_median=st.median(obj),
               obj_min=min(obj), obj_max=max(obj), held_mean=st.mean(held), held_sd=st.stdev(held) if len(held) > 1 else 0.0,
               held_median=st.median(held), L_values=" ".join(str(s["L"]) for s in g), lam_values=" ".join(f"{fl(s['lam_best']):.0e}" for s in g),
               seconds_mean=st.mean(fl(s["seconds"]) for s in g))
    optima.append(row)
cols = list(optima[0].keys())
for r in optima:
    for k in r:
        if k not in cols: cols.append(k)
with open(RES / "optima.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(optima)

# running minimum curves
with open(RES / "running_min.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["mode", "N_R", "set", "task", "eval_index", "mean_running_min", "median_running_min", "n_searches"])
    for key in sorted(running, key=lambda k: (k[0], k[1], k[2], TASKS.index(k[3]))):
        curves = running[key]; n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        for i in range(n):
            w.writerow([*key, i + 1, np.nanmean(arr[:, i]), np.nanmedian(arr[:, i]), len(curves)])

# S0 enumeration
with open(RES / "s0.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["mode", "N_R", "task", "L", "objective_mean", "objective_sd_over_seeds", "held_mean", "n_seeds", "lam_values"])
    grp = defaultdict(list)
    for r in s0rows:
        grp[(r["mode"], r["N_R"], r["task"], r["L"])].append(r)
    for key in sorted(grp, key=lambda k: (k[0], k[1], TASKS.index(k[2]), k[3])):
        g = grp[key]; o = [r["objective"] for r in g]
        w.writerow([*key, st.mean(o), st.stdev(o) if len(o) > 1 else 0.0, st.mean(r["held"] for r in g), len(g), " ".join(f"{r['lam_best']:.0e}" for r in g)])

print(f"{len(searches)} searches, {len(optima)} optimum rows, {len(s0rows)} S0 rows")
for r in optima:
    print(f"{r['mode']:3s} NR={r['N_R']:6d} {r['set']} {r['task']:8s} n={r['n_searches']:2d} best {fl(r['objective']):.4f} (held {fl(r['held']):.4f}) mean {r['obj_mean']:.4f} sd {r['obj_sd']:.4f} | L {r['L_values']} | lam {r['lam_values']} | koffT {fl(r['koff_T']):.2f} cpk/KD {fl(r['c_peak_over_KD']):.2f} tauD/T {fl(r['tau_D_over_T']):.3f}")
