"""Fresh re-evaluation of the additional-task optima (XOR, PAT, NARMA) and their input-tap baselines.
Mean-field S0/S1/S4 optima: readout retrained under noise at 500 ... 500000 (R realizations); noise-aware optima: designed
readout at their design N_R. Usage: python reevaluate_additional_tasks.py <R> <n_workers>   (the manuscript used R = 50)
Outputs: data/results/additional_tasks_and_readouts/reevaluation.csv and baseline_taps.csv"""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import simulator as mc
RES = mc.RESULTS / "additional_tasks_and_readouts"; R = int(sys.argv[1]); NW = int(sys.argv[2])
NRS = [500, 2000, 5000, 20000, 50000, 200000, 500000]


def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))
def fl(x):
    try: return float(x)
    except Exception: return float("nan")


def job(a):
    o, NR, proto, k = a
    p = mc.params_from_row(o); task = o["task"]; seed = 80000 + 3 * k
    base = dict(kind="bits", group="bits", design_mode=o["mode"], design_NR=int(o["N_R"]), set=o["set"], design_task=task, eval_task=task, eval_NR=NR, protocol=proto, R=R)
    if NR == 0:
        b = mc.evaluate(task, p, mode="det", lams=(fl(o["lam_best"]),)); return dict(base, L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=0.0, held=b["held"])
    if proto == "designed":
        r = mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=(fl(o["lam_best"]),)); return dict(base, L=r["L"], lam=r["lam_best"], objective=r["objective"], opt_sd=r["opt_sd"], held=r["held"])
    b = mc.best_over_L(mc.evaluate_all_L(task, p, mode="hyb", N_R=NR, R=R, seed=seed)); return dict(base, L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=b["opt_sd"], held=b["held"])


if __name__ == "__main__":
    optima = rd(RES / "optima.csv"); jobs = []; k = 0
    for o in optima:
        if o["mode"] == "det":
            jobs.append((o, 0, "designed", k)); k += 1
            for NR in NRS: jobs.append((o, NR, "retrained", k)); k += 1
        else:
            jobs.append((o, int(o["N_R"]), "designed", k)); k += 1
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r)
            if i % 10 == 0: print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f} s", flush=True)
    with open(RES / "reevaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = [mc.input_tap_baseline(t, taps) for t in ("XOR", "PAT", "NARMA") for taps in (1, 2, 3, 5, 10, 20, 50)]
    with open(RES / "baseline_taps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base[0].keys())); w.writeheader(); w.writerows(base)
    print("wrote reevaluation.csv and baseline_taps.csv", f"{time.time()-t0:.0f} s")
