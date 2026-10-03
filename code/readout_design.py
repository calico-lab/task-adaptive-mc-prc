"""Readout designed for noise: readout variants evaluated on the same realizations of the receptor process.

Variants (all applied to the same R realizations of the receptor process, simulated once):
  integrating nodes with fewer, longer windows   M' in a divisor set of M; node value = mean occupancy over T/M'
  causal low-pass (exponential moving average)   time constant tau_f, applied to the node sequence at the node rate
  longer readout memory                          L up to 20
  ridge parameter selected from the grid of simulator.py (simulator.LAMBDAS)
evaluate_variants(task, params, N_R, R, seed, variants) returns one row per variant (Mp, tau_f_over_T, L) with the
lambda-selected objective and held-out value; the row 'sampled_default' (M sampled nodes, no low-pass, the L of the
design) is the default readout retrained on the same realizations.
evaluate_readout(...) evaluates one variant (used by the joint channel-plus-readout search).
"""
from __future__ import annotations
import math
import numpy as np
import simulator as mc

M_OPTIONS = {50: (50, 25, 10, 5), 100: (100, 50, 20, 10)}
TAU_OPTIONS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)   # in units of T
L_OPTIONS = (5, 10, 20)


class _Proxy:
    def __init__(self, M, n): self.base_nodes, self.num_tot_points = M, n


def reduce_nodes(integ: np.ndarray, M: int, Mp: int, n_sym: int) -> np.ndarray:
    """Means over M/Mp consecutive integrating nodes (exact, equal windows)."""
    if Mp == M: return integ
    g = M // Mp
    return integ.reshape(n_sym, Mp, g).mean(axis=2).reshape(-1)


def ema_nodes(nodes: np.ndarray, a: float) -> np.ndarray:
    if a <= 0.0: return nodes
    out = np.empty_like(nodes); y = nodes[0]; out[0] = y
    for j in range(1, nodes.size):
        y = a * y + (1.0 - a) * nodes[j]; out[j] = y
    return out


def _node_sets(task, params, mode, N_R, R, seed):
    td = mc.get_task(task)
    g, c, counts = mc.concentration(td, params)
    n_nodes = td.num_tot_points * g["M"]
    if mode == "det":
        samp, integ = mc._meanfield_nodes(c, g["dt"], params["k_on"], params["k_off"], g["n_sub"], n_nodes)
        return td, g, [samp], [integ]
    samp, integ = mc._markov_nodes(c, g["dt"], params["k_on"], params["k_off"], int(N_R), int(R), int(seed), g["n_sub"], n_nodes, mc.NORMAL_MEAN_THRESHOLD)
    return td, g, list(samp), list(integ)


def _score_variant(td, node_list, M, Mp, tau_over_T, L, lams, sampled=False):
    per = {lam: dict(opt=[], held=[]) for lam in lams}
    a = math.exp(-(1.0 / Mp) / tau_over_T) if tau_over_T > 0 else 0.0
    for nd in node_list:
        red = nd if sampled else reduce_nodes(nd, M, Mp, td.num_tot_points)
        red = ema_nodes(red, a)
        X = mc.states_from_nodes(_Proxy(Mp if not sampled else M, td.num_tot_points), red, L)
        sc = mc.fit_scores(td, X, lams)
        for lam in lams:
            per[lam]["opt"].append(sc[lam]["opt"]); per[lam]["held"].append(sc[lam]["held"])
    om = {lam: float(np.mean(per[lam]["opt"])) for lam in lams}
    lb = min(lams, key=lambda l: om[l] if np.isfinite(om[l]) else np.inf)
    R = len(node_list)
    return dict(lam=lb, objective=om[lb], opt_sd=float(np.std(per[lb]["opt"], ddof=1)) if R > 1 else 0.0, held=float(np.mean(per[lb]["held"])))


def evaluate_variants(task, params, N_R, R, seed, Ms=None, taus=TAU_OPTIONS, Ls=L_OPTIONS, lams=mc.LAMBDAS, mode="hyb"):
    td, g, samp, integ = _node_sets(task, params, mode, N_R, R, seed)
    M = g["M"]; Ms = Ms or M_OPTIONS[M]
    rows = []
    base = _score_variant(td, samp, M, M, 0.0, int(params["L"]), lams, sampled=True)
    rows.append(dict(task=task, N_R=N_R, R=R, variant="sampled_default", Mp=M, tau_over_T=0.0, L=int(params["L"]), **base))
    for Mp in Ms:
        for tau in taus:
            for L in Ls:
                if Mp * L > 1000: continue
                r = _score_variant(td, integ, M, Mp, tau, L, lams)
                rows.append(dict(task=task, N_R=N_R, R=R, variant="integrating", Mp=Mp, tau_over_T=tau, L=L, **r))
    return rows


def evaluate_readout(task, params, Mp, tau_over_T, L, mode="hyb", N_R=500, R=10, seed=0, lams=mc.LAMBDAS):
    td, g, samp, integ = _node_sets(task, params, mode, N_R, R, seed)
    r = _score_variant(td, integ, g["M"], Mp, tau_over_T, L, lams)
    r.update(task=task, N_R=N_R, R=R, Mp=Mp, tau_over_T=tau_over_T, L=L)
    return r
