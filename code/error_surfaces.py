"""Error surfaces of the benchmark tasks.

det   receiver grid (k_on, k_off), transmitter grid (T, N_max), channel grid (d, D), each at the nominal values of the
      other parameters, best L in 1..5 and lambda selected at every grid point (mean field)
hyb   receiver grid under receptor-count noise at a given N_R (R realizations, fresh seed per point), coarser grid
Usage: python error_surfaces.py det <n_workers> [n_grid]
       python error_surfaces.py hyb <n_workers> <N_R> <R> [n_grid]
The manuscript used the default grids, and the hyb mode at N_R = 500 and 50000 with R = 10.
Outputs in data/results/benchmark_tasks: error_surface_receiver.csv, error_surface_transmitter.csv,
error_surface_channel.csv, error_surface_receiver_hyb_NR<N_R>.csv
"""
import csv, sys, time, math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import simulator as mc

RES = mc.RESULTS / "benchmark_tasks"; MODE = sys.argv[1]; NW = int(sys.argv[2])
FIELDS = ["task", "k_on", "k_off", "T", "distance", "N_max", "D", "L_best", "lam_best", "nrmse_opt_best", "nrmse_held_best", "opt_sd"] + [f"nrmse_opt_L{L}" for L in range(1, 6)] + \
         ["KD_nM", "koff_T", "c_peak_over_KD", "c_mean_over_KD", "tau_D_over_T", "phi_ISI", "occ_mean", "occ_max", "occ_frac_above_0p8", "contrast_within", "binomial_sd_mean", "N_R", "R"]


def point(args):
    task, params, mode, NR, R, seed = args
    try:
        allL = mc.evaluate_all_L(task, params, mode=mode, N_R=NR, R=R, seed=seed)
        b = mc.best_over_L(allL)
        row = dict(task=task, **{k: params[k] for k in mc.PARAM_NAMES}, L_best=b["L"], lam_best=b["lam_best"], nrmse_opt_best=b["objective"], nrmse_held_best=b["held"], opt_sd=b["opt_sd"])
        for L in range(1, 6):
            row[f"nrmse_opt_L{L}"] = allL[L]["objective"]
        for k in FIELDS:
            if k not in row and k in b:
                row[k] = b[k]
        row["N_R"], row["R"] = (NR if mode == "hyb" else 0), (R if mode == "hyb" else 1)
    except Exception as e:
        row = dict(task=task, **{k: params[k] for k in mc.PARAM_NAMES}, L_best=5, lam_best=float("nan"), nrmse_opt_best=float("nan"), nrmse_held_best=float("nan"), opt_sd=float("nan"), N_R=NR, R=R)
    return row


def grid(name, xa, ya, n):
    kx, lox, hix = mc.DOMAIN[xa]; ky, loy, hiy = mc.DOMAIN[ya]
    xs = np.logspace(math.log10(lox), math.log10(hix), n) if kx == "log" else np.linspace(lox, hix, n)
    ys = np.logspace(math.log10(loy), math.log10(hiy), n) if ky == "log" else np.linspace(loy, hiy, n)
    return [(x, y) for x in xs for y in ys]


def run(name, xa, ya, n, mode, NR, R, out):
    jobs = []
    for task in mc.TASKS:
        for i, (x, y) in enumerate(grid(name, xa, ya, n)):
            params = dict(mc.NOMINAL, L=5); params[xa] = float(x); params[ya] = float(y)
            jobs.append((task, params, mode, NR, R, 20000 + i))
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for i, r in enumerate(ex.map(point, jobs, chunksize=4)):
            rows.append(r)
            if i % 200 == 0: print(f"{name}: {i+1}/{len(jobs)} {time.time()-t0:.0f} s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("wrote", out, f"{time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    if MODE == "det":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        run("receiver", "k_on", "k_off", n, "det", 0, 1, RES / "error_surface_receiver.csv")
        run("transmitter", "T", "N_max", n, "det", 0, 1, RES / "error_surface_transmitter.csv")
        run("channel", "distance", "D", 21, "det", 0, 1, RES / "error_surface_channel.csv")
    else:
        NR = int(sys.argv[3]); R = int(sys.argv[4]); n = int(sys.argv[5]) if len(sys.argv) > 5 else 17
        run("receiver", "k_on", "k_off", n, "hyb", NR, R, RES / f"error_surface_receiver_hyb_NR{NR}.csv")
