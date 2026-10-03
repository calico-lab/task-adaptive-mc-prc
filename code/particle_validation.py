"""Stochastic receptor model against the particle-based simulations (12 configurations, five Smoldyn realizations each).

The receptor Markov chain of simulator.py is driven by the mean-field concentration of each configuration on the
simulator grid, R realizations with fresh seeds. Every realization is processed exactly as the Smoldyn records were
(particle_processing.py): occupancy resampled to the 10 ms record grid, trailing moving average over K+1 samples
(K = 2000 for MG tasks, 6000 for sine-to-square), virtual nodes at rounded indices of the 10 ms grid, ridge lambda 1e-6
with a penalized intercept, NRMSE on the optimization block; the raw score uses the same pipeline without the average.
In addition every realization is scored with the readout rule of simulator.py (nodes on the grid, L and lambda
selected; columns hybrid_retrained*) to bridge the two protocols.
Usage: python particle_validation.py <R>   (the manuscript used R = 50)
Inputs: data/particle_simulations/smoldyn_selected_candidates_12.csv and smoldyn_60_seed_level_values.csv
Output: data/results/benchmark_tasks/particle_validation.csv
"""
import csv, sys, time, statistics as st
import numpy as np
import simulator as mc
import particle_processing as pp

RES = mc.RESULTS / "benchmark_tasks"; R = int(sys.argv[1])
K_FILTER = {"MG": 2000, "MGCUBED": 2000, "SINE": 6000}
DT_PART = 0.01


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def smoldyn_processing(td, cfg, occ10, time10, K, filtered):
    s = np.convolve(occ10, np.ones(K + 1) / (K + 1), mode="full")[: occ10.size] if filtered else occ10
    states = pp.build_states(td, dict(cfg, dt=DT_PART), dict(time=time10, occupation=s), cfg["memorywindowlength"], indexing="rounded")
    return pp.score(td, states, lam=1e-6)["opt"]


sel = rd(mc.ROOT / "data" / "particle_simulations" / "smoldyn_selected_candidates_12.csv")
smol = rd(mc.ROOT / "data" / "particle_simulations" / "smoldyn_60_seed_level_values.csv")
rows = []
for i, r in enumerate(sel):
    task, p = r["task"], mc.params_from_row(r)
    td = mc.get_task(task); cfg = pp.make_cfg(r)
    t0 = time.time()
    g, c, counts = mc.concentration(td, p)
    nb = mc.markov_full(c, g["dt"], p["k_on"], p["k_off"], 500, R, 9000 + i)
    time_grid = np.arange(g["n_time"]) * g["dt"]
    time10 = np.arange(0.0, time_grid[-1] + 1e-9, DT_PART)
    idx = np.clip(np.round(time10 / g["dt"]).astype(int), 0, g["n_time"] - 1)
    raws, filts = [], []
    for k in range(R):
        occ10 = nb[k, idx] / 500.0
        raws.append(smoldyn_processing(td, cfg, occ10, time10, K_FILTER[task], False))
        filts.append(smoldyn_processing(td, cfg, occ10, time10, K_FILTER[task], True))
    det10 = mc.meanfield_full(c, g["dt"], p["k_on"], p["k_off"])[idx]
    det_raw = smoldyn_processing(td, cfg, det10, time10, K_FILTER[task], False)
    det_filt = smoldyn_processing(td, cfg, det10, time10, K_FILTER[task], True)
    # readout rule of the simulator on the same chain (retrained readout)
    ret = mc.best_over_L(mc.evaluate_all_L(task, p, mode="hyb", N_R=500, R=min(R, 20), seed=9500 + i))
    sm = [x for x in smol if x["selection_id"] == r["selection_id"]]
    sr = [float(x["raw_smoldyn_nrmse"]) for x in sm]; sf = [float(x["filtered_smoldyn_nrmse"]) for x in sm]
    row = dict(selection_id=r["selection_id"], set=r["scenario"], task=task, archived_bo_nrmse=float(r["bo_nrmse"]), meanfield=mc.evaluate(task, p, lams=(1e-6,))["objective"],
               det_10ms_raw=det_raw, det_10ms_filtered=det_filt,
               hybrid_raw_mean=st.mean(raws), hybrid_raw_sd=st.stdev(raws), hybrid_filt_mean=st.mean(filts), hybrid_filt_sd=st.stdev(filts),
               smoldyn_raw_mean=st.mean(sr), smoldyn_raw_sd=st.stdev(sr), smoldyn_filt_mean=st.mean(sf), smoldyn_filt_sd=st.stdev(sf), n_smoldyn=len(sf),
               hybrid_retrained=ret["objective"], hybrid_retrained_sd=ret["opt_sd"], hybrid_retrained_L=ret["L"], hybrid_retrained_lam=ret["lam_best"],
               max_bound_mean=float(nb.max(axis=1).mean()), R=R, seconds=time.time() - t0)
    rows.append(row)
    print(f"{r['selection_id']:12s} | Smoldyn raw {row['smoldyn_raw_mean']:.3f}±{row['smoldyn_raw_sd']:.3f} filt {row['smoldyn_filt_mean']:.3f}±{row['smoldyn_filt_sd']:.3f} | hybrid raw {row['hybrid_raw_mean']:.3f}±{row['hybrid_raw_sd']:.3f} filt {row['hybrid_filt_mean']:.3f}±{row['hybrid_filt_sd']:.3f} | retrained {row['hybrid_retrained']:.3f} (L={ret['L']}, lam={ret['lam_best']:.0e}) | {row['seconds']:.0f} s", flush=True)
with open(RES / "particle_validation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", RES / "particle_validation.csv")
