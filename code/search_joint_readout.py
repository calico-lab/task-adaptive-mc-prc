"""Joint noise-aware search of the channel and the readout (one search per process).
Channel parameters of the set (S1 receptor rates; S3 receptor rates, symbol interval up to 5 s, release amplitude) plus
the readout variants of readout_design.py (node count M' categorical, low-pass time constant tau/T log-uniform in
[0.05, 5], readout memory L in 1..20), ridge selected inside. Objective: mean NRMSE over R realizations at N_R receptors
(fixed chain seed). The search starts from 12 Latin-hypercube points. launch_joint_readout.py runs the searches of the
manuscript.
python search_joint_readout.py --set S1|S3 --task MG --NR 50000 --R 10 --seed 1 --outdir <dir>
"""
import argparse, csv, json, sys, time, math
from pathlib import Path
import numpy as np
import readout_design as m
mc = m.mc
ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["S1", "S3"]); ap.add_argument("--task", required=True); ap.add_argument("--NR", type=int, default=50000)
ap.add_argument("--R", type=int, default=10); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--n-calls", type=int, default=100); ap.add_argument("--outdir", required=True)
a = ap.parse_args()
sub = Path(a.outdir) / f"readout_hyb_NR{a.NR}"; sub.mkdir(parents=True, exist_ok=True)
stem = f"{a.set}_{a.task}_seed{a.seed}"; csv_path, done_path = sub / f"{stem}.csv", sub / f"{stem}.done"
if done_path.exists(): print("already done"); sys.exit(0)
if csv_path.exists(): csv_path.unlink()
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
M = mc.get_task(a.task).base_nodes
free = ["k_on", "k_off"] + (["T", "N_max"] if a.set == "S3" else [])
space = [Real(5e-20, 2e-17, prior="log-uniform", name="k_on"), Real(0.1, 10.0, prior="log-uniform", name="k_off")]
if a.set == "S3": space += [Real(0.5, 5.0, name="T"), Real(200.0, 20000.0, name="N_max")]
space += [Categorical(list(m.M_OPTIONS[M]), name="Mp"), Real(0.05, 5.0, prior="log-uniform", name="tau_over_T"), Integer(1, 20, name="L")]
chain_seed = 1000 + a.seed; best = [math.inf, None]; counter = [0]; t_start = time.time()


def objective(x):
    counter[0] += 1
    params = dict(mc.NOMINAL)
    for name, v in zip(free, x[: len(free)]): params[name] = float(v)
    Mp, tau, L = int(x[len(free)]), float(x[len(free) + 1]), int(x[len(free) + 2]); params["L"] = L
    try:
        r = m.evaluate_readout(a.task, params, Mp, tau, L, mode="hyb", N_R=a.NR, R=a.R, seed=chain_seed)
        ok = np.isfinite(r["objective"])
    except Exception as e:
        r = dict(objective=float("nan"), error=str(e)[:200], lam=float("nan"), opt_sd=float("nan"), held=float("nan")); ok = False
    row = dict(set=a.set, task=a.task, N_R=a.NR, R=a.R, search_seed=a.seed, eval_index=counter[0], is_finite=int(ok), **{k: params[k] for k in mc.PARAM_NAMES}, Mp=Mp, tau_over_T=tau, L=L,
               lam=r.get("lam"), objective=r["objective"], opt_sd=r.get("opt_sd"), held=r.get("held"))
    if ok and r["objective"] < best[0]: best[0], best[1] = r["objective"], row
    row["running_min"] = best[0] if math.isfinite(best[0]) else float("nan")
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)
    return r["objective"] if ok else 5.0


gp_minimize(objective, space, n_calls=a.n_calls, n_initial_points=12, initial_point_generator="lhs", acq_func="EI", noise="gaussian", random_state=a.seed, n_jobs=1)
done_path.write_text(json.dumps(dict(set=a.set, task=a.task, N_R=a.NR, R=a.R, search_seed=a.seed, chain_seed=chain_seed, seconds=time.time() - t_start,
                                     best={k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in (best[1] or {}).items()}), indent=1, default=str))
print(f"{stem} NR={a.NR}: best {best[0]:.4f} in {(time.time()-t_start)/60:.1f} min")
