"""Re-evaluation of the case-study, wide-domain, and minimax optima (run after collect_case_study_minimax_wide_domain.py).
1. Fresh re-evaluation of the case-study optima (designed readout, R realizations) at their design N_R, and of the
   mean-field case-study optima under noise with the readout retrained, at 500 and 50000 receptors.
2. Input-tap baselines for the case-study tasks.
3. Fresh re-evaluation of the wide-domain optima (noise-aware ones at 50000) and of the compromise optima on every task.
Usage: python reevaluate_case_study_minimax_wide_domain.py <R> <n_workers>   (the manuscript used R = 50)
Outputs: data/results/case_study_minimax_wide_domain/reevaluation.csv and baseline_taps.csv
"""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import simulator_extensions as m
mc = m.mc
RES = mc.RESULTS / "case_study_minimax_wide_domain"
R = int(sys.argv[1]); NW = int(sys.argv[2])


def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def fl(x):
    try: return float(x)
    except Exception: return float("nan")


def job(args):
    kind, o, task, NR, proto, k = args
    m.use_domain("wide" if o["group"] == "wide" else "default")
    p = mc.params_from_row(o); seed = 40000 + 3 * k
    base = dict(kind=kind, group=o["group"], design_mode=o["mode"], design_NR=int(o["N_R"]), set=o["set"], design_task=o["task"], eval_task=task, eval_NR=NR, protocol=proto, R=R)
    if NR == 0:
        allL = mc.evaluate_all_L(task, p, mode="det"); b = mc.best_over_L(allL) if proto == "retrained" else mc.evaluate(task, p, mode="det", lams=(fl(o["lam_best"]),))
        return dict(base, L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=0.0, held=b["held"])
    if proto == "designed":
        r = mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=(fl(o["lam_best"]),))
        return dict(base, L=r["L"], lam=r["lam_best"], objective=r["objective"], opt_sd=r["opt_sd"], held=r["held"])
    b = mc.best_over_L(mc.evaluate_all_L(task, p, mode="hyb", N_R=NR, R=R, seed=seed))
    return dict(base, L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=b["opt_sd"], held=b["held"])


if __name__ == "__main__":
    optima = rd(RES / "optima.csv"); jobs = []; k = 0
    for o in optima:
        if o["group"] == "case":
            if o["mode"] == "hyb":
                jobs.append(("case", o, o["task"], int(o["N_R"]), "designed", k)); k += 1
            else:
                for NR in (0, 500, 50000):
                    jobs.append(("case", o, o["task"], NR, "designed" if NR == 0 else "retrained", k)); k += 1
        elif o["group"] == "wide":
            NR = int(o["N_R"]) if o["mode"] == "hyb" else 0
            jobs.append(("wide", o, o["task"], NR, "designed", k)); k += 1
        elif o["group"] == "compromise":
            for t in ("MG", "SINE", "MGCUBED"):
                NR = int(o["N_R"]) if o["mode"] == "hyb" else 0
                jobs.append(("compromise", o, t, NR, "retrained" if NR else "retrained", k)); k += 1
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r)
            if i % 10 == 0: print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f} s", flush=True)
    with open(RES / "reevaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = []
    for t in ("GLUF", "GLUE"):
        for taps in (1, 2, 3, 5, 10, 20, 50): base.append(mc.input_tap_baseline(t, taps))
    with open(RES / "baseline_taps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base[0].keys())); w.writeheader(); w.writerows(base)
    print("wrote reevaluation.csv and baseline_taps.csv", f"{time.time()-t0:.0f} s")
