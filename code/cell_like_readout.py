"""Cell-like readout. One or two integrating windows per symbol (M' = 1, 2), a causal low-pass on the node sequence,
and a readout memory of 5, 10, or 20 symbols, for the S1 and S4 mean-field optima of the eight tasks, under receptor-count
noise (R realizations at 500, 5000, 50000, 500000 receptors) and in the mean field (N_R = 0). Same grid, ridge rule, and
blocks as readout_grid.py; the row 'sampled_default' is the default readout on the same realizations.
Usage: python cell_like_readout.py <R> <n_workers>   (the manuscript used R = 20)
Output: data/results/cell_like_readout_and_esn/cell_like_readout.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import readout_design as m  # noqa: E402
mc = m.mc
OUT = mc.RESULTS / "cell_like_readout_and_esn" / "cell_like_readout.csv"
R = int(sys.argv[1]); NW = int(sys.argv[2]); NRS = [500, 5000, 50000, 500000]
FILES = [mc.RESULTS / "benchmark_tasks" / "optima.csv", mc.RESULTS / "case_study_minimax_wide_domain" / "optima.csv",
         mc.RESULTS / "additional_tasks_and_readouts" / "optima.csv"]


def rd(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def job(a):
    o, NR, k = a
    p = mc.params_from_row(o); det = NR == 0
    rows = m.evaluate_variants(o["task"], p, 0 if det else NR, 1 if det else R, 80000 + 5 * k, Ms=(2, 1), mode="det" if det else "hyb")
    for r in rows: r.update(set=o["set"], group=o.get("group", "benchmark"))
    return rows


if __name__ == "__main__":
    optima = [r for f in FILES for r in rd(f) if r["mode"] == "det" and r["set"] in ("S1", "S4") and r.get("group", "benchmark") in ("benchmark", "case", "bits")]
    jobs = [(o, NR, k) for k, (o, NR) in enumerate((o, NR) for o in optima for NR in [0] + NRS)]
    print(len(optima), "optima,", len(jobs), "jobs", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists(): OUT.unlink()
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, rr in enumerate(ex.map(job, jobs)):
            write_header = not OUT.exists()
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
                if write_header: w.writeheader()
                w.writerows(rr)
            n += len(rr)
            base = [r for r in rr if r["variant"] == "sampled_default"][0]
            one = min([r for r in rr if r["variant"] == "integrating" and int(r["Mp"]) == 1], key=lambda r: r["objective"])
            two = min([r for r in rr if r["variant"] == "integrating" and int(r["Mp"]) == 2], key=lambda r: r["objective"])
            print(f"{i+1}/{len(jobs)} {rr[0]['set']} {rr[0]['task']:8s} NR={rr[0]['N_R']:>7d} default {base['objective']:.3f} | M'=1 best {one['objective']:.3f} (tau/T {one['tau_over_T']}, L {one['L']}) | M'=2 best {two['objective']:.3f} {time.time()-t0:.0f} s", flush=True)
    print("wrote", n, "rows to", OUT, flush=True)
