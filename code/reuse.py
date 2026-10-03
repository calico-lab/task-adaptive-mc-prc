"""Cross-task reuse of the benchmark-task optima with the readout retrained (L in 1..5 and lambda re-selected on the target task).

Mean field: for every set and every ordered task pair, the best configuration of every search for the source task
(searches.csv) is evaluated on the target task; make_numbers.py computes the reuse penalties from these rows as ratios
of medians over searches.
Under noise: for every design N_R and set, the noise-aware optimum for the source task is evaluated on the target task at
the same N_R with R realizations and fresh chain seeds; the reference is the target's own noise-aware optimum evaluated
with the same protocol (same R, fresh seeds).
Usage: python reuse.py <R> <n_workers>   (the manuscript used R = 20)
Output: data/results/benchmark_tasks/reuse.csv
"""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import simulator as mc

RES = mc.RESULTS / "benchmark_tasks"; R = int(sys.argv[1]); NW = int(sys.argv[2])
TASKS = ("MG", "SINE", "MGCUBED")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def det_job(args):
    src_row, target = args
    p = mc.params_from_row(src_row)
    allL = mc.evaluate_all_L(target, p, mode="det"); b = mc.best_over_L(allL)
    return dict(mode="det", N_R=0, set=src_row["set"], source_task=src_row["task"], target_task=target, search_seed=int(src_row["search_seed"]),
                L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=0.0, held=b["held"], R=1, chain_seed=0)


def hyb_job(args):
    src_row, target, k = args
    p = mc.params_from_row(src_row); NR = int(src_row["N_R"]); seed = 7000 + 13 * k
    allL = mc.evaluate_all_L(target, p, mode="hyb", N_R=NR, R=R, seed=seed); b = mc.best_over_L(allL)
    return dict(mode="hyb", N_R=NR, set=src_row["set"], source_task=src_row["task"], target_task=target, search_seed=int(src_row["search_seed"]),
                L=b["L"], lam=b["lam_best"], objective=b["objective"], opt_sd=b["opt_sd"], held=b["held"], R=R, chain_seed=seed)


if __name__ == "__main__":
    searches = rd(RES / "searches.csv"); optima = rd(RES / "optima.csv")
    det_jobs = [(s, t) for s in searches if s["mode"] == "det" and s["set"] != "S0" for t in TASKS]
    hyb_jobs = []; k = 0
    for o in optima:
        if o["mode"] == "hyb" and o["set"] != "S0":
            for t in TASKS:
                hyb_jobs.append((o, t, k)); k += 1
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        rows.extend(ex.map(det_job, det_jobs))
        print(f"det reuse done: {len(rows)} rows, {time.time()-t0:.0f} s", flush=True)
        for i, r in enumerate(ex.map(hyb_job, hyb_jobs)):
            rows.append(r)
            if i % 10 == 0: print(f"hyb reuse {i+1}/{len(hyb_jobs)}, {time.time()-t0:.0f} s", flush=True)
    with open(RES / "reuse.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", RES / "reuse.csv", len(rows), "rows")
