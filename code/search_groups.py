"""Bayesian-optimization search for the case study, the wide domain, the minimax objective, and the additional tasks
(one search per process).

python search_groups.py --group case|wide|compromise|bits --set S1 --tasks GLUF --mode det|hyb --NR 50000 --R 10 --seed 1
                        --outdir <dir>

group case        case-study tasks (GLUF, GLUE), default domain
group wide        benchmark tasks, wide domain (simulator_extensions.DOMAIN_WIDE)
group compromise  several tasks at once (--tasks MG,SINE,MGCUBED); objective = max over tasks of NRMSE_t / ref_t, where
                  ref_t is the task-specific optimum of the same set (and N_R) in data/results/benchmark_tasks/optima.csv;
                  its optimum is the minimax configuration of the manuscript
group bits        additional tasks (XOR, PAT, NARMA), default domain
Search space and objective otherwise as in search_benchmark_tasks.py. In mode det, --NR enters only the noise
descriptors of the output rows (binomial_sd_mean, snr_within); the launchers pass --NR 0 for the mean-field searches.
Every evaluation is appended to <outdir>/<group>_<mode>[_NR<NR>]/<set>_<tasklabel>_seed<seed>.csv; a .done file marks
completion. launch_case_study_minimax_wide_domain.py and launch_additional_tasks.py run the searches of the manuscript.
"""
import argparse, csv, json, sys, time, math
from pathlib import Path
import numpy as np
import simulator_extensions as m
mc = m.mc

ap = argparse.ArgumentParser()
ap.add_argument("--group", required=True, choices=["case", "wide", "compromise", "bits"]); ap.add_argument("--set", required=True)
ap.add_argument("--tasks", required=True); ap.add_argument("--mode", default="det", choices=["det", "hyb"]); ap.add_argument("--NR", type=int, default=50000)
ap.add_argument("--R", type=int, default=10); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--n-calls", type=int, default=100)
ap.add_argument("--n-initial", type=int, default=10); ap.add_argument("--outdir", required=True)
a = ap.parse_args()
tasks = a.tasks.split(",")
m.use_domain("wide" if a.group == "wide" else "default")
NR = a.NR if a.mode == "hyb" else 0
sub = Path(a.outdir) / (f"{a.group}_{a.mode}" + (f"_NR{NR}" if a.mode == "hyb" else ""))
sub.mkdir(parents=True, exist_ok=True)
label = "+".join(tasks)
stem = f"{a.set}_{label}_seed{a.seed}"
csv_path, done_path = sub / f"{stem}.csv", sub / f"{stem}.done"
if done_path.exists():
    print("already done", done_path); sys.exit(0)
if csv_path.exists():
    csv_path.unlink()
chain_seed = 1000 + a.seed
free = list(mc.SETS[a.set])
t_start = time.time(); best = [math.inf, None]

refs = {}
if a.group == "compromise":
    opt = list(csv.DictReader(open(mc.RESULTS / "benchmark_tasks" / "optima.csv", encoding="utf-8-sig")))
    for t in tasks:
        rows = [r for r in opt if r["set"] == a.set and r["task"] == t and r["mode"] == a.mode and int(float(r["N_R"])) == NR]
        if not rows: sys.exit(f"no reference optimum for {a.set} {t} {a.mode} {NR}")
        refs[t] = float(rows[0]["objective"])


def write_row(row):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)


def run_point(params, idx):
    params = dict(params)
    try:
        if a.group == "compromise":
            obj, rows, ratios = m.compromise_objective(params, tasks, refs, a.mode, a.NR, a.R, chain_seed)
            r = dict(rows[tasks[0]]); r["task"] = label; r["objective"] = obj; r["held"] = max(rows[t]["held"] / refs[t] for t in tasks)
            for t in tasks:
                r[f"obj_{t}"] = rows[t]["objective"]; r[f"ratio_{t}"] = ratios[t]; r[f"lam_{t}"] = rows[t]["lam_best"]; r[f"held_{t}"] = rows[t]["held"]; r[f"ref_{t}"] = refs[t]
        else:
            r = mc.evaluate(tasks[0], params, mode=a.mode, N_R=a.NR, R=a.R, seed=chain_seed)
        ok = np.isfinite(r["objective"])
    except Exception as e:
        r = dict(task=label, mode=a.mode, N_R=NR, R=a.R, seed=chain_seed, **{k: params[k] for k in mc.PARAM_NAMES}, L=int(params["L"]), objective=float("nan"), error=str(e)[:200]); ok = False
    row = dict(group=a.group, set=a.set, search_seed=a.seed, eval_index=idx, is_finite=int(ok)); row.update(r)
    if ok and r["objective"] < best[0]: best[0], best[1] = r["objective"], row
    row["running_min"] = best[0] if math.isfinite(best[0]) else float("nan")
    write_row(row)
    return r["objective"] if ok else 5.0


if not free:
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
        for name, v in zip(free, x[:-1]): params[name] = float(v)
        params["L"] = int(x[-1])
        return run_point(params, counter[0])

    gp_minimize(objective, space, n_calls=a.n_calls, n_initial_points=a.n_initial, initial_point_generator="lhs",
                acq_func="EI", noise="gaussian" if a.mode == "hyb" else 1e-10, random_state=a.seed, n_jobs=1)

summary = dict(group=a.group, set=a.set, task=label, tasks=tasks, mode=a.mode, N_R=NR, R=a.R if a.mode == "hyb" else 1, search_seed=a.seed, chain_seed=chain_seed,
               domain=("wide" if a.group == "wide" else "default"), refs=refs, n_evaluations=len(open(csv_path).readlines()) - 1, seconds=time.time() - t_start,
               best={k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in (best[1] or {}).items()})
done_path.write_text(json.dumps(summary, indent=1, default=str))
print(f"{a.group} {stem} {a.mode} NR={NR}: best {best[0]:.4f} in {summary['seconds']/60:.1f} min")
