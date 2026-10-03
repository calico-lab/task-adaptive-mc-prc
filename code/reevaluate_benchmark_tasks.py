"""Re-evaluation of the benchmark-task optima under receptor-count noise, fresh chain seeds, several receptor numbers.

For every optimum in optima.csv (mean-field optima of S0 to S4, noise-aware optima of S1 to S4 at each design N_R)
and every N_R in the list, two protocols:
  designed   readout as designed (L and lambda of the optimum), channel as designed
  retrained  channel as designed, readout retrained under noise (L in 1..5 and lambda re-selected on the optimization block)
Each protocol uses R realizations with chain seeds independent of the search seeds. Mean-field ("det") values of every
optimum are recorded too. Also the integrating-node variant of the retrained protocol.
Usage: python reevaluate_benchmark_tasks.py <R> <n_workers> [NR list comma separated]
The manuscript used R = 50 and the default NR list (500 to 500000 receptors).
Output: data/results/benchmark_tasks/reevaluation.csv
"""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import simulator as mc

RES = mc.RESULTS / "benchmark_tasks"; R = int(sys.argv[1]); NW = int(sys.argv[2])
NRS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [500, 2000, 5000, 20000, 50000, 200000, 500000]
OUT = RES / "reevaluation.csv"


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def job(args):
    opt, NR, k = args
    p = mc.params_from_row(opt)
    task = opt["task"]
    seed = 5000 + 17 * k
    rows = []
    base = dict(design_mode=opt["mode"], design_NR=int(opt["N_R"]), set=opt["set"], task=task, design_objective=float(opt["objective"]),
                design_L=p["L"], design_lam=float(opt["lam_best"]), eval_NR=NR, R=R, chain_seed=seed)
    # designed readout
    r = mc.evaluate(task, p, mode="hyb", N_R=NR, R=R, seed=seed, lams=(float(opt["lam_best"]),))
    rows.append(dict(base, protocol="designed", node="sample", L=r["L"], lam=r["lam_best"], objective=r["objective"], opt_sd=r["opt_sd"], held=r["held"], held_sd=r["held_sd"]))
    # retrained readout (L and lambda), sampled nodes
    allL = mc.evaluate_all_L(task, p, mode="hyb", N_R=NR, R=R, seed=seed)
    b = mc.best_over_L(allL)
    rows.append(dict(base, protocol="retrained", node="sample", L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=b["opt_sd"], held=b["held"], held_sd=b["held_sd"],
                     **{f"obj_L{L}": allL[L]["objective"] for L in allL}))
    # retrained readout, integrating nodes
    allI = mc.evaluate_all_L(task, p, mode="hyb", N_R=NR, R=R, seed=seed, node="integrate")
    bi = mc.best_over_L(allI)
    rows.append(dict(base, protocol="retrained", node="integrate", L=bi["L"], lam=bi["lam_best"], objective=bi["objective"], opt_sd=bi["opt_sd"], held=bi["held"], held_sd=bi["held_sd"]))
    return rows


if __name__ == "__main__":
    optima = rd(RES / "optima.csv")
    jobs = []
    k = 0
    for opt in optima:
        for NR in NRS:
            jobs.append((opt, NR, k)); k += 1
    # mean-field values of every optimum (fast, serial)
    det_rows = []
    for opt in optima:
        p = mc.params_from_row(opt)
        r = mc.evaluate(opt["task"], p, mode="det", lams=(float(opt["lam_best"]),))
        allL = mc.evaluate_all_L(opt["task"], p, mode="det"); b = mc.best_over_L(allL)
        base = dict(design_mode=opt["mode"], design_NR=int(opt["N_R"]), set=opt["set"], task=opt["task"], design_objective=float(opt["objective"]),
                    design_L=p["L"], design_lam=float(opt["lam_best"]), eval_NR=0, R=1, chain_seed=0)
        det_rows.append(dict(base, protocol="designed", node="sample", L=r["L"], lam=r["lam_best"], objective=r["objective"], opt_sd=0.0, held=r["held"], held_sd=0.0))
        det_rows.append(dict(base, protocol="retrained", node="sample", L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=0.0, held=b["held"], held_sd=0.0,
                             **{f"obj_L{L}": allL[L]["objective"] for L in allL}))
    t0 = time.time()
    results = list(det_rows)
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, rows in enumerate(ex.map(job, jobs)):
            results.extend(rows)
            if i % 10 == 0:
                print(f"{i+1}/{len(jobs)} jobs, {time.time()-t0:.0f} s", flush=True)
    cols = []
    for r in results:
        for c in r:
            if c not in cols: cols.append(c)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    print("wrote", OUT, len(results), "rows", f"{time.time()-t0:.0f} s")
