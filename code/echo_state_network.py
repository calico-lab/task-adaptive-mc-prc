"""Reference echo state network of equal size. For each task an ESN with N = 5 M leaky tanh units (250 for the
tasks with M = 50 virtual nodes, 500 for sine-to-square), readout on the state, the scaled input, and a constant, with the
ridge grid, the blocks, and the lambda rule of the channel readout. Spectral radius, input scaling, leak rate, and bias
scaling are tuned by Bayesian optimization with the budget of the channel searches (100 evaluations, 10 Latin-hypercube
initial points, expected improvement), three searches per task with the weight seed fixed within a search. The best
configuration of each search is re-evaluated with 10 fresh weight seeds.
Usage: python echo_state_network.py <n_workers>
Output: data/results/cell_like_readout_and_esn/echo_state_network.csv (one row per search)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import simulator as mc  # noqa: E402
from skopt import gp_minimize  # noqa: E402
from skopt.space import Real  # noqa: E402
OUT = mc.RESULTS / "cell_like_readout_and_esn" / "echo_state_network.csv"
TASKS = ["MG", "SINE", "MGCUBED", "XOR", "PAT", "NARMA", "GLUF", "GLUE"]
SPACE = [Real(0.1, 1.5, name="rho"), Real(0.01, 10.0, prior="log-uniform", name="s_in"), Real(0.05, 1.0, name="leak"), Real(0.0, 1.0, name="s_b")]
N_CALLS, N_INIT, N_SEARCH, FRESH, DENSITY = 100, 10, 3, 10, 0.1


def scaled_input(td):
    u = td.input_series.astype(float)
    if td.task == "SINE": return u
    return (u - u.min()) / (u.max() - u.min())


def esn_states(u, N, rho, s_in, leak, s_b, seed):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1.0, 1.0, (N, N)) * (rng.random((N, N)) < DENSITY)
    ev = float(np.max(np.abs(np.linalg.eigvals(W))))
    if ev > 0: W *= rho / ev
    w_in = rng.uniform(-1.0, 1.0, N) * s_in; b = rng.uniform(-1.0, 1.0, N) * s_b
    X = np.zeros((N, u.size)); x = np.zeros(N)
    for n in range(u.size):
        x = (1.0 - leak) * x + leak * np.tanh(W @ x + w_in * u[n] + b)
        X[:, n] = x
    return X


def score(task, params, seed):
    td = mc.get_task(task); u = scaled_input(td); N = 5 * td.base_nodes
    X = esn_states(u, N, *params, seed)
    F = np.vstack([X, u[None, :]])
    sc = mc.fit_scores(td, F, mc.LAMBDAS)
    lb = min(mc.LAMBDAS, key=lambda l: sc[l]["opt"] if np.isfinite(sc[l]["opt"]) else np.inf)
    return dict(lam=lb, opt=float(sc[lb]["opt"]), held=float(sc[lb]["held"]), train=float(sc[lb]["train"]), N=N)


def search(a):
    task, s = a; t0 = time.time()
    def progress(r):
        k = len(r.x_iters)
        if k % 25 == 0: print(f"  {task} seed {s}: {k} calls, best {r.fun:.4f}, {time.time()-t0:.0f} s", flush=True)
    res = gp_minimize(lambda p: float(min(score(task, p, 1000 + s)["opt"], 5.0)), SPACE, n_calls=N_CALLS, n_initial_points=N_INIT, initial_point_generator="lhs", acq_func="EI", random_state=s, callback=[progress])
    best = [float(v) for v in res.x]
    fresh = [score(task, best, 5000 + i) for i in range(FRESH)]
    opts = np.array([f["opt"] for f in fresh]); helds = np.array([f["held"] for f in fresh])
    return dict(task=task, search_seed=s, N=fresh[0]["N"], rho=best[0], s_in=best[1], leak=best[2], s_b=best[3], search_objective=float(res.fun),
                fresh_mean=float(opts.mean()), fresh_sd=float(opts.std(ddof=1)), held_mean=float(helds.mean()), held_sd=float(helds.std(ddof=1)),
                lam=fresh[0]["lam"], seconds=time.time() - t0)


if __name__ == "__main__":
    NW = int(sys.argv[1]); jobs = [(t, s) for t in TASKS for s in range(1, N_SEARCH + 1)]
    OUT.parent.mkdir(parents=True, exist_ok=True); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for r in ex.map(search, jobs):
            rows.append(r); print(f"{r['task']:8s} seed {r['search_seed']} N={r['N']} search {r['search_objective']:.4f} fresh {r['fresh_mean']:.4f} +- {r['fresh_sd']:.4f} held {r['held_mean']:.4f} rho {r['rho']:.2f} s_in {r['s_in']:.3f} leak {r['leak']:.2f} s_b {r['s_b']:.2f} {r['seconds']:.0f} s", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", OUT, flush=True)
