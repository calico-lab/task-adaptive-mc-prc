"""Collect the searches on the additional tasks (group bits, search_groups.py) and the joint channel-plus-readout searches
(search_joint_readout.py). Writes searches.csv and optima.csv (additional tasks), joint_readout_searches.csv, and
joint_readout_optima.csv into data/results/additional_tasks_and_readouts.
Usage: python collect_additional_tasks_and_readouts.py"""
import csv, glob, json, statistics as st
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RES = ROOT / "data" / "results" / "additional_tasks_and_readouts"


def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))
def fl(x):
    try: return float(x)
    except Exception: return float("nan")


def collect(pattern, keyfun, out_searches, out_optima, skip=("group", "set", "search_seed", "eval_index", "is_finite", "mode", "N_R", "task", "running_min", "seed")):
    searches = []
    for done in sorted(glob.glob(str(RES / "trials" / pattern / "*.done"))):
        info = json.load(open(done)); rows = rd(done.replace(".done", ".csv")); fin = [r for r in rows if r["is_finite"] == "1"]
        if not fin: continue
        best = min(fin, key=lambda r: fl(r["objective"]))
        out = dict(group=info.get("group", "readout"), mode=info.get("mode", "hyb"), N_R=int(info["N_R"]), set=info["set"], task=info["task"], search_seed=int(info["search_seed"]), n_evaluations=len(rows), seconds=fl(info["seconds"]))
        for k, v in best.items():
            if k not in skip: out[k] = v
        searches.append(out)
    if not searches: return
    cols = []
    for s in searches:
        for k in s:
            if k not in cols: cols.append(k)
    with open(RES / out_searches, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(searches)
    groups = defaultdict(list)
    for s in searches: groups[keyfun(s)].append(s)
    optima = []
    for key, g in sorted(groups.items()):
        obj = [fl(s["objective"]) for s in g]; best = min(g, key=lambda s: fl(s["objective"])); row = dict(best)
        row.update(n_searches=len(g), obj_mean=st.mean(obj), obj_sd=st.stdev(obj) if len(obj) > 1 else 0.0, obj_min=min(obj), L_values=" ".join(str(s["L"]) for s in g))
        optima.append(row)
    cols = []
    for r in optima:
        for k in r:
            if k not in cols: cols.append(k)
    with open(RES / out_optima, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(optima)
    print(out_optima, len(searches), "searches,", len(optima), "optima")
    for r in optima:
        extra = f" Mp {r['Mp']} tau/T {fl(r['tau_over_T']):.2f}" if "Mp" in r else f" koffT {fl(r.get('koff_T', 'nan')):.2f} cpk/KD {fl(r.get('c_peak_over_KD', 'nan')):.2f}"
        print(f"  {r['group']:8s} {r['mode']:3s} NR={r['N_R']:6d} {r['set']} {r['task']:8s} n={r['n_searches']} best {fl(r['objective']):.4f} mean {r['obj_mean']:.4f} L {r['L_values']}{extra}")


if __name__ == "__main__":
    collect("bits_*", lambda s: (s["group"], s["mode"], s["N_R"], s["set"], s["task"]), "searches.csv", "optima.csv")
    collect("readout_*", lambda s: (s["mode"], s["N_R"], s["set"], s["task"]), "joint_readout_searches.csv", "joint_readout_optima.csv")
