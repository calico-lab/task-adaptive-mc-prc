"""Collect the searches of the case study, the wide domain, and the minimax (compromise) objective into searches.csv
and optima.csv in data/results/case_study_minimax_wide_domain.
Usage: python collect_case_study_minimax_wide_domain.py"""
import csv, glob, json, statistics as st
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
RES = ROOT / "data" / "results" / "case_study_minimax_wide_domain"


def fl(x):
    try: return float(x)
    except Exception: return float("nan")


searches = []
for done in sorted(glob.glob(str(RES / "trials" / "*" / "*.done"))):
    info = json.load(open(done)); rows = list(csv.DictReader(open(done.replace(".done", ".csv"), encoding="utf-8-sig")))
    fin = [r for r in rows if r["is_finite"] == "1"]
    if not fin: continue
    best = min(fin, key=lambda r: fl(r["objective"]))
    out = dict(group=info["group"], mode=info["mode"], N_R=int(info["N_R"]), set=info["set"], task=info["task"], search_seed=int(info["search_seed"]),
               domain=info.get("domain", "default"), n_evaluations=len(rows), seconds=fl(info["seconds"]))
    for k, v in best.items():
        if k not in ("group", "set", "search_seed", "eval_index", "is_finite", "mode", "N_R", "task", "running_min", "seed"): out[k] = v
    searches.append(out)
cols = []
for s in searches:
    for k in s:
        if k not in cols: cols.append(k)
with open(RES / "searches.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(searches)
groups = defaultdict(list)
for s in searches: groups[(s["group"], s["mode"], s["N_R"], s["set"], s["task"])].append(s)
optima = []
for key, g in sorted(groups.items()):
    obj = [fl(s["objective"]) for s in g]; best = min(g, key=lambda s: fl(s["objective"]))
    row = dict(best); row.update(n_searches=len(g), obj_mean=st.mean(obj), obj_sd=st.stdev(obj) if len(obj) > 1 else 0.0, obj_min=min(obj), obj_max=max(obj),
                                 L_values=" ".join(str(s["L"]) for s in g), lam_values=" ".join(f"{fl(s['lam_best']):.0e}" for s in g))
    optima.append(row)
cols = []
for r in optima:
    for k in r:
        if k not in cols: cols.append(k)
with open(RES / "optima.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(optima)
print(len(searches), "searches,", len(optima), "optimum rows")
for r in optima:
    extra = " ".join(f"{k}={fl(r[k]):.3f}" for k in r if k.startswith("obj_") and k[4:] in ("MG", "SINE", "MGCUBED"))
    print(f"{r['group']:10s} {r['mode']:3s} NR={r['N_R']:6d} {r['set']} {r['task']:16s} n={r['n_searches']} best {fl(r['objective']):.4f} mean {r['obj_mean']:.4f} | L {r['L_values']} lam {r['lam_values']} | koffT {fl(r['koff_T']):.2f} cpk/KD {fl(r['c_peak_over_KD']):.2f} tauD/T {fl(r['tau_D_over_T']):.3f} T {fl(r['T']):.2f} d {1e6*fl(r['distance']):.1f} {extra}")
