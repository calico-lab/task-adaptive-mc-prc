"""Baselines: ridge regression on the raw input history (no channel) with 1, 2, 3, 5, 10, 20, and 50 taps, and the optima
with L = 1.
Usage: python baselines_benchmark_tasks.py [R]   (default R = 20)
Outputs: data/results/benchmark_tasks/baseline_taps.csv and baseline_L1.csv
"""
import csv, sys
import simulator as mc

RES = mc.RESULTS / "benchmark_tasks"; R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
for task in mc.TASKS:
    for taps in (1, 2, 3, 5, 10, 20, 50):
        rows.append(mc.input_tap_baseline(task, taps))
with open(RES / "baseline_taps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['task']:8s} taps {r['taps']:2d} | opt {r['objective']:.4f} held {r['held']:.4f} lam {r['lam_best']:.0e}")
# optima with L = 1 (channel kept, lambda re-selected), mean field and under noise at the design N_R (hyb) or 500 (det)
optima = list(csv.DictReader(open(RES / "optima.csv", encoding="utf-8-sig")))
rows = []
for i, o in enumerate(optima):
    p = mc.params_from_row(o); p["L"] = 1
    d = mc.evaluate(o["task"], p, mode="det")
    NR = int(o["N_R"]) if o["mode"] == "hyb" else 500
    h = mc.evaluate(o["task"], p, mode="hyb", N_R=NR, R=R, seed=11000 + i)
    rows.append(dict(design_mode=o["mode"], design_NR=int(o["N_R"]), set=o["set"], task=o["task"], design_objective=float(o["objective"]), design_L=int(float(o["L"])),
                     det_L1=d["objective"], det_L1_lam=d["lam_best"], det_L1_held=d["held"], hyb_NR=NR, hyb_L1=h["objective"], hyb_L1_sd=h["opt_sd"], hyb_L1_lam=h["lam_best"], hyb_L1_held=h["held"]))
    print(f"{o['mode']:3s} NR={int(o['N_R']):6d} {o['set']} {o['task']:8s} design {float(o['objective']):.4f} (L={int(float(o['L']))}) | L=1 det {d['objective']:.4f} | L=1 hyb@{NR} {h['objective']:.4f}")
with open(RES / "baseline_L1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
