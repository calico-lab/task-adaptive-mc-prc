"""Numerical robustness of the benchmark-task optima (parallel).
For every optimum in optima.csv: mean-field objective under (a) the default grid (dt <= 1 ms), (b) a four times finer
grid, (c) the full concentration history instead of round(100 tau_D/T) symbols; and under noise (R realizations, same
seed) the same three variants plus (d) the exact binomial sampler instead of the normal approximation above mean 25.
Usage: python robustness.py <R> [n_workers]   (the manuscript used R = 20)
Output: data/results/benchmark_tasks/robustness.csv
"""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import simulator as mc

RES = mc.RESULTS / "benchmark_tasks"; R = int(sys.argv[1]); NW = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def job(args):
    i, o = args
    p = mc.params_from_row(o); task = o["task"]; lam = (float(o["lam_best"]),)
    NR = int(o["N_R"]) if o["mode"] == "hyb" else 500
    td = mc.get_task(task); full_hist = td.num_tot_points
    det = dict(default=mc.evaluate(task, p, mode="det", lams=lam)["objective"], fine=mc.evaluate(task, p, mode="det", lams=lam, dt_max=0.25e-3)["objective"],
               full_history=mc.evaluate(task, p, mode="det", lams=lam, history=full_hist)["objective"])
    seed = 30000 + i
    hyb = dict(default=mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=lam), fine=mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=lam, dt_max=0.25e-3),
               full_history=mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=lam, history=full_hist))
    mc.EXACT_BINOMIAL = True
    hyb["exact_binomial"] = mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=lam)
    mc.EXACT_BINOMIAL = False
    return dict(design_mode=o["mode"], design_NR=int(o["N_R"]), set=o["set"], task=task, L=p["L"], lam=lam[0], eval_NR=NR, R=R,
                det_default=det["default"], det_fine=det["fine"], det_full_history=det["full_history"],
                hyb_default=hyb["default"]["objective"], hyb_default_sd=hyb["default"]["opt_sd"], hyb_fine=hyb["fine"]["objective"], hyb_full_history=hyb["full_history"]["objective"],
                hyb_exact_binomial=hyb["exact_binomial"]["objective"], hyb_exact_binomial_sd=hyb["exact_binomial"]["opt_sd"], history_symbols=int(float(o["history_symbols"])), dt_ms=1e3 * float(o["dt"]))


if __name__ == "__main__":
    optima = list(csv.DictReader(open(RES / "optima.csv", encoding="utf-8-sig")))
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for row in ex.map(job, list(enumerate(optima))):
            rows.append(row)
            print(f"{row['design_mode']:3s} NR={row['design_NR']:6d} {row['set']} {row['task']:8s} | det {row['det_default']:.4f} fine {row['det_fine']:.4f} fullhist {row['det_full_history']:.4f} | hyb {row['hyb_default']:.4f} fine {row['hyb_fine']:.4f} fullhist {row['hyb_full_history']:.4f} exact {row['hyb_exact_binomial']:.4f} | {time.time()-t0:.0f} s", flush=True)
    with open(RES / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", RES / "robustness.csv")
