"""Bayesian-optimization search on one benchmark task (one search per process).

python search_benchmark_tasks.py --set S1 --task MG --mode det|hyb --NR 500 --R 10 --seed 1 --n-calls 100 --outdir <dir>

Search space: the parameters of the tunable-parameter set (simulator.SETS, bounds simulator.DOMAIN, log-uniform for
k_on, k_off, distance, D) plus the integer readout length L in 1..5. The other parameters stay at the nominal
values. Objective: mean NRMSE on the optimization block (mode det: mean field; mode hyb: mean over R receptor
Markov-chain realizations with N_R receptors, fixed chain seed within a search). The ridge parameter is
chosen inside every evaluation (smallest optimization-block error over simulator.LAMBDAS).
S0 has no free channel parameter and is evaluated directly for L = 1..5. In mode det, --NR enters only the noise
descriptors of the output rows (binomial_sd_mean, snr_within); the launchers keep the default of 500.
Every evaluation is appended to <outdir>/hyb_NR<NR>/<set>_<task>_seed<seed>.csv (mode hyb) or to
<outdir>/det/<set>_<task>_seed<seed>.csv (mode det) as it completes; a .done file with the best row marks completion
and lets the launcher skip finished searches. launch_benchmark_tasks.py and launch_benchmark_tasks_extra.py run
the searches of the manuscript with <outdir> = data/results/benchmark_tasks/trials.
"""
import argparse, csv, json, sys, time, math
from pathlib import Path
import numpy as np
import simulator as mc

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True); ap.add_argument("--task", required=True)
ap.add_argument("--mode", default="det", choices=["det", "hyb"]); ap.add_argument("--NR", type=int, default=500)
ap.add_argument("--R", type=int, default=10); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--n-calls", type=int, default=100); ap.add_argument("--n-initial", type=int, default=10)
ap.add_argument("--node", default="sample"); ap.add_argument("--outdir", required=True)
a = ap.parse_args()

NR = a.NR if a.mode == "hyb" else 0
sub = Path(a.outdir) / (f"{a.mode}_NR{NR}" if a.mode == "hyb" else "det")
sub.mkdir(parents=True, exist_ok=True)
stem = f"{a.set}_{a.task}_seed{a.seed}"
csv_path, done_path = sub / f"{stem}.csv", sub / f"{stem}.done"
if done_path.exists():
    print("already done", done_path); sys.exit(0)
if csv_path.exists():
    csv_path.unlink()
chain_seed = 1000 + a.seed
free = list(mc.SETS[a.set])
t_start = time.time()
rows = []
best = [math.inf, None]


def write_row(row):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run_point(params, idx):
    params = dict(params)
    try:
        r = mc.evaluate(a.task, params, mode=a.mode, N_R=a.NR, R=a.R, seed=chain_seed, node=a.node)
        ok = np.isfinite(r["objective"])
    except Exception as e:  # record the failure, keep the search alive
        r = dict(task=a.task, mode=a.mode, node=a.node, N_R=NR, R=a.R, seed=chain_seed, **{k: params[k] for k in mc.PARAM_NAMES}, L=int(params["L"]), objective=float("nan"), error=str(e)[:200])
        ok = False
    row = dict(set=a.set, search_seed=a.seed, eval_index=idx, is_finite=int(ok))
    row.update(r)
    if ok and r["objective"] < best[0]:
        best[0], best[1] = r["objective"], row
    row["running_min"] = best[0] if math.isfinite(best[0]) else float("nan")
    write_row(row)
    return r["objective"] if ok else 2.0


if not free:  # S0: readout length only
    for idx, L in enumerate(range(1, 6), start=1):
        run_point(dict(mc.NOMINAL, L=L), idx)
else:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    space = []
    for name in free:
        kind, lo, hi = mc.DOMAIN[name]
        space.append(Real(lo, hi, prior="log-uniform" if kind == "log" else "uniform", name=name))
    space.append(Integer(1, 5, name="L"))
    counter = [0]

    def objective(x):
        counter[0] += 1
        params = dict(mc.NOMINAL)
        for name, v in zip(free, x[:-1]):
            params[name] = float(v)
        params["L"] = int(x[-1])
        return run_point(params, counter[0])

    gp_minimize(objective, space, n_calls=a.n_calls, n_initial_points=a.n_initial, initial_point_generator="lhs",
                acq_func="EI", noise="gaussian" if a.mode == "hyb" else 1e-10, random_state=a.seed, n_jobs=1)

summary = dict(set=a.set, task=a.task, mode=a.mode, N_R=NR, R=a.R if a.mode == "hyb" else 1, search_seed=a.seed, chain_seed=chain_seed,
               n_evaluations=len(open(csv_path).readlines()) - 1, seconds=time.time() - t_start,
               best={k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in (best[1] or {}).items()})
done_path.write_text(json.dumps(summary, indent=1, default=str))
print(f"{stem} {a.mode} NR={NR}: best {best[0]:.4f} in {summary['seconds']/60:.1f} min")
