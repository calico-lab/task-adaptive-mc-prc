"""Linear memory capacity and second-order capacity against receptor number (Supporting Information).

Protocol (Dambre et al. 2012, restricted). Input u(n) i.i.d. uniform on [0, 1], 2200 symbols, encoded as in the tasks.
State from the configuration's own parameters with M = 50 nodes and its own L. Readout by ridge regression with the
ridge value selected on the scoring rows, fitted on rows 501 to 1500 and scored on rows 1501 to 2200 by the squared
correlation between prediction and target. Linear capacity = sum over delays 1..30 of r^2 for u(n-k); second-order
capacity = sum over k = 0..15 of r^2 for P2(2u(n-k)-1) plus sum over pairs 0 <= k1 < k2 <= 5 of r^2 for the product
(2u(n-k1)-1)(2u(n-k2)-1). Mean field (N_R = 0 in the table) and stochastic model with R realizations at each N_R.
Configurations: the mean-field S1 optimum of each benchmark task and the nominal configuration.
Usage: python capacity.py <R> [N_R list comma separated]
The manuscript used R = 5, i.e., 15 realizations per row over the three input seeds.
Output: data/results/case_study_minimax_wide_domain/capacity.csv
"""
import csv, sys, time
import numpy as np
import simulator as mc

OUT = mc.RESULTS / "case_study_minimax_wide_domain" / "capacity.csv"; R = int(sys.argv[1])
NRS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [500, 5000, 50000, 500000]
N_SYM, TRAIN, TEST = 2200, np.arange(500, 1500), np.arange(1500, 2200)
K_LIN, K_QUAD, K_CROSS, SEEDS = 30, 15, 5, (11, 22, 33)


class UTask:
    def __init__(self, u):
        self.base_nodes, self.num_tot_points, self.input_series, self.task = 50, N_SYM, u, "capacity"


def r2(pred, target):
    if np.std(pred) < 1e-12 or np.std(target) < 1e-12: return 0.0
    return float(np.corrcoef(pred, target)[0, 1] ** 2)


def fit_score(X, targets):
    """Best-ridge r^2 on the test rows for every target (targets: dict name -> series)."""
    Xtr = np.vstack([X[:, TRAIN], np.ones((1, TRAIN.size))]); Xte = np.vstack([X[:, TEST], np.ones((1, TEST.size))])
    G = Xtr @ Xtr.T; evals, V = np.linalg.eigh(G); evals = np.maximum(evals, 0)
    out = {}
    for name, y in targets.items():
        VtXy = V.T @ (Xtr @ y[TRAIN]); best = 0.0
        for lam in mc.LAMBDAS:
            w = V @ (VtXy / (evals + lam)); best = max(best, r2(Xte.T @ w, y[TEST]))
        out[name] = best
    return out


def capacities(params, N_R, R, seed_in):
    rng = np.random.default_rng(seed_in); u = rng.random(N_SYM); td = UTask(u)
    g, c, counts = mc.concentration(td, params); n_nodes = N_SYM * 50; L = int(params["L"])
    targets = {}
    for k in range(1, K_LIN + 1): targets[f"lin{k}"] = np.concatenate([np.zeros(k), u[:-k]])
    z = 2 * u - 1
    for k in range(0, K_QUAD + 1): zk = np.concatenate([np.zeros(k), z[:N_SYM - k]]); targets[f"quad{k}"] = 0.5 * (3 * zk ** 2 - 1)
    for k1 in range(0, K_CROSS + 1):
        for k2 in range(k1 + 1, K_CROSS + 1):
            z1 = np.concatenate([np.zeros(k1), z[:N_SYM - k1]]); z2 = np.concatenate([np.zeros(k2), z[:N_SYM - k2]]); targets[f"cross{k1}_{k2}"] = z1 * z2
    if N_R == 0:
        samp, _ = mc._meanfield_nodes(c, g["dt"], params["k_on"], params["k_off"], g["n_sub"], n_nodes); node_sets = [samp]
    else:
        samp, _ = mc._markov_nodes(c, g["dt"], params["k_on"], params["k_off"], int(N_R), int(R), int(seed_in + 7), g["n_sub"], n_nodes, mc.NORMAL_MEAN_THRESHOLD); node_sets = list(samp)
    lin, quad, prof = [], [], []
    for nd in node_sets:
        sc = fit_score(mc.states_from_nodes(td, nd, L), targets)
        lin.append(sum(v for k, v in sc.items() if k.startswith("lin"))); quad.append(sum(v for k, v in sc.items() if not k.startswith("lin")))
        prof.append([sc[f"lin{k}"] for k in range(1, 11)] + [sc[f"quad{k}"] for k in range(0, 6)])
    return np.array(lin), np.array(quad), np.array(prof)


if __name__ == "__main__":
    optima = list(csv.DictReader(open(mc.RESULTS / "benchmark_tasks" / "optima.csv", encoding="utf-8-sig")))
    configs = {"nominal": dict(mc.NOMINAL, L=5)}
    for t in mc.TASKS:
        o = [r for r in optima if r["mode"] == "det" and r["set"] == "S1" and r["task"] == t][0]; configs[f"S1_{t}"] = mc.params_from_row(o)
    rows = []; t0 = time.time()
    for name, p in configs.items():
        for NR in [0] + NRS:
            lins, quads, profs = [], [], []
            for s in SEEDS:
                l, q, pr = capacities(p, NR, R if NR else 1, s); lins += list(l); quads += list(q); profs += list(pr)
            prof = np.mean(np.array(profs), axis=0)
            row = dict(config=name, N_R=NR, L=p["L"], linear_capacity=np.mean(lins), linear_capacity_sd=np.std(lins, ddof=1), quadratic_capacity=np.mean(quads), quadratic_capacity_sd=np.std(quads, ddof=1), n=len(lins))
            for k in range(1, 11): row[f"lin_k{k}"] = prof[k - 1]
            for k in range(0, 6): row[f"quad_k{k}"] = prof[10 + k]
            rows.append(row)
            print(f"{name:12s} N_R={NR:7d} linear {np.mean(lins):6.2f} +- {np.std(lins, ddof=1):.2f}  quadratic {np.mean(quads):6.2f} +- {np.std(quads, ddof=1):.2f}  {time.time()-t0:.0f} s", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", OUT)
