"""Extensions of the simulator: the wide search domain, time-varying receptor rates (task switching in operation),
the compromise (minimax) objective over several tasks, and the case-study tasks GLUF and GLUE.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
import simulator as mc

DOMAIN_WIDE = dict(k_on=("log", 5e-20, 2e-17), k_off=("log", 0.1, 100.0), T=("lin", 0.5, 5.0),
                   N_max=("lin", 200.0, 1e5), distance=("log", 3.5e-6, 50e-6), D=("log", 5e-12, 2e-10))
CASE_TASKS = ("GLUF", "GLUE")
ALL_TASKS = mc.TASKS + CASE_TASKS
TNAME = {"MG": "MG forecasting", "SINE": "sine-to-square", "MGCUBED": "MG-Cubed", "GLUF": "biomarker forecast", "GLUE": "biomarker episode"}


def use_domain(name: str):
    mc.DOMAIN.clear()
    mc.DOMAIN.update(DOMAIN_WIDE if name == "wide" else dict(k_on=("log", 5e-20, 2e-17), k_off=("log", 0.1, 10.0), T=("lin", 0.5, 2.0),
                                                              N_max=("lin", 200.0, 20000.0), distance=("log", 6e-6, 50e-6), D=("log", 5e-12, 2e-10)))


# ----------------------------------------------------------------------------- time-varying receptor rates
@njit(cache=True)
def _meanfield_nodes_tv(c, dt, kon_t, koff_t, n_sub, n_nodes):
    n_time = c.size
    samp = np.zeros(n_nodes)
    x = 0.0
    for t in range(1, n_time):
        r = kon_t[t - 1] * c[t - 1] + koff_t[t - 1]
        beq = kon_t[t - 1] * c[t - 1] / r
        x = beq + (x - beq) * math.exp(-r * dt)
        if t % n_sub == 0:
            j = t // n_sub
            if j < n_nodes:
                samp[j] = x
    return samp


@njit(cache=True)
def _markov_nodes_tv(c, dt, kon_t, koff_t, NR, R, seed, n_sub, n_nodes, thr):
    np.random.seed(seed)
    n_time = c.size
    samp = np.zeros((R, n_nodes))
    state = np.zeros(R, dtype=np.int64)
    for t in range(1, n_time):
        r = kon_t[t - 1] * c[t - 1] + koff_t[t - 1]
        beq = kon_t[t - 1] * c[t - 1] / r
        q = 1.0 - math.exp(-r * dt)
        p_fb = beq * q
        p_bf = (1.0 - beq) * q
        for i in range(R):
            s = state[i]
            s = s + mc._binom(NR - s, p_fb, thr) - mc._binom(s, p_bf, thr)
            state[i] = s
        if t % n_sub == 0:
            j = t // n_sub
            if j < n_nodes:
                for i in range(R):
                    samp[i, j] = state[i] / NR
    return samp


class LongTask:
    """Task-like object for a long record (switching runs): input and two targets over n symbols.
    Pairs that share an input (MG/MGCUBED, GLUF/GLUE) use the shared series; other pairs concatenate the first half
    of the source input with the second half of the target input (the task changes entirely at the switch).
    base_nodes is the least common multiple of the two node counts, and readouts subsample the node sequence."""
    def __init__(self, task_a: str, task_b: str, n_symbols: int):
        self.task = f"{task_a}->{task_b}"
        self.num_tot_points = n_symbols
        Ma, Mb = mc.get_task(task_a).base_nodes, mc.get_task(task_b).base_nodes
        self.nodes_per_task = {task_a: Ma, task_b: Mb}
        self.base_nodes = int(np.lcm(Ma, Mb))
        shared = {task_a, task_b} <= set(CASE_TASKS) or {task_a, task_b} <= {"MG", "MGCUBED"}
        if not shared:
            half = n_symbols // 2
            ta, tb = mc.get_task(task_a), mc.get_task(task_b)
            self.input_series = np.concatenate([ta.input_series[:half], tb.input_series[:n_symbols - half]])
            self.targets = {task_a: np.concatenate([ta.target_series[:half], np.zeros(n_symbols - half)]),
                            task_b: np.concatenate([np.zeros(half), tb.target_series[:n_symbols - half]])}
            self.target_sd = {task_a: np.std(ta.target_series[:half], ddof=1), task_b: np.std(tb.target_series[:n_symbols - half], ddof=1)}
            return
        if task_a in CASE_TASKS:
            d = np.load(mc.DATA / "glucose_case_study.npz", allow_pickle=True)
            self.input_series = d["input"][:n_symbols].astype(float).copy()
            self.targets = {"GLUF": d["target_forecast"][:n_symbols].astype(float).copy(), "GLUE": d["target_episode"][:n_symbols].astype(float).copy()}
        else:
            from task_data import load_mat
            series = load_mat(mc.DATA / "MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat")["mackey_glass_series"].reshape(-1)
            seg = series[: n_symbols + 6]
            norm = (seg - seg.min()) / (seg.max() - seg.min())
            mgc = load_mat(mc.DATA / "MGCubed_series_k10.mat")
            self.input_series = norm[:-6].copy()
            self.targets = {"MG": norm[6:].copy(), "MGCUBED": mgc["target_series"].reshape(-1)[:n_symbols].copy()}
        if abs(self.input_series.max() - 1) > 1e-9 or self.input_series.min() < -1e-9:
            self.input_series = (self.input_series - self.input_series.min()) / (self.input_series.max() - self.input_series.min())
        self.target_sd = {k: np.std(v, ddof=1) for k, v in self.targets.items()}


class _Proxy:
    def __init__(self, M, n): self.base_nodes, self.num_tot_points = M, n


def states_for_task(lt: LongTask, nodes: np.ndarray, task: str, L: int) -> np.ndarray:
    """State matrix for a readout trained with the node count of `task`, from the node sequence of the long run."""
    M = lt.nodes_per_task[task]; step = lt.base_nodes // M
    nd = nodes.reshape(-1, lt.base_nodes)[:, ::step].reshape(-1) if step > 1 else nodes
    return mc.states_from_nodes(_Proxy(M, lt.num_tot_points), nd, L)


def rate_schedule(g: dict, n_symbols: int, pa: dict, pb: dict, n_switch: int, ramp: int):
    """k_on(t), k_off(t) on the grid: pa before symbol n_switch, log-linear ramp over ramp symbols, pb after."""
    sps, n_time = g["steps_per_symbol"], g["n_time"]
    t_sym = np.arange(n_time) / sps
    f = np.clip((t_sym - n_switch) / max(ramp, 1e-9), 0.0, 1.0) if ramp > 0 else (t_sym >= n_switch).astype(float)
    kon = np.exp(np.log(pa["k_on"]) + f * (np.log(pb["k_on"]) - np.log(pa["k_on"])))
    koff = np.exp(np.log(pa["k_off"]) + f * (np.log(pb["k_off"]) - np.log(pa["k_off"])))
    return kon, koff


def train_readout(task: str, params: dict, N_R: int, seed: int, lam: float, mode: str = "hyb"):
    """Readout weights of a standalone run of the standard task (training block), one realization."""
    td = mc.get_task(task)
    g, c, counts = mc.concentration(td, params)
    n_nodes = td.num_tot_points * g["M"]
    if mode == "det":
        nodes, _ = mc._meanfield_nodes(c, g["dt"], params["k_on"], params["k_off"], g["n_sub"], n_nodes)
    else:
        samp, _ = mc._markov_nodes(c, g["dt"], params["k_on"], params["k_off"], int(N_R), 1, int(seed), g["n_sub"], n_nodes, mc.NORMAL_MEAN_THRESHOLD)
        nodes = samp[0]
    L = int(round(params["L"]))
    X = mc.states_from_nodes(td, nodes, L)
    bl = mc.block_slices(td)
    Xtr = np.vstack([X[:, bl["train"]], np.ones((1, bl["train"].size))])
    w = np.linalg.solve(Xtr @ Xtr.T + lam * np.eye(Xtr.shape[0]), Xtr @ td.target_series[bl["train"]])
    return w, L


def switching_run(lt: LongTask, task_a: str, task_b: str, pa: dict, pb: dict, wa, La, wb, Lb, n_switch: int, ramp: int,
                  N_R: int, R: int, seed: int, mode: str = "hyb", schedule: str = "switch"):
    """Predictions of the long run for R realizations. schedule 'switch' (A then ramp to B), 'A' (A throughout), 'B' (B throughout).
    Returns dict with per-realization prediction arrays for target A (readout wa) and target B (readout wb) at every symbol."""
    shared = dict(pa); shared["L"] = La
    g, c, counts = mc.concentration(lt, shared)  # transmitter and channel parameters are those of pa (S1 switching keeps them)
    n_nodes = lt.num_tot_points * g["M"]
    if schedule == "switch":
        kon, koff = rate_schedule(g, lt.num_tot_points, pa, pb, n_switch, ramp)
    elif schedule == "A":
        kon = np.full(g["n_time"], pa["k_on"]); koff = np.full(g["n_time"], pa["k_off"])
    else:
        kon = np.full(g["n_time"], pb["k_on"]); koff = np.full(g["n_time"], pb["k_off"])
    if mode == "det":
        node_sets = [_meanfield_nodes_tv(c, g["dt"], kon, koff, g["n_sub"], n_nodes)]
    else:
        node_sets = list(_markov_nodes_tv(c, g["dt"], kon, koff, int(N_R), int(R), int(seed), g["n_sub"], n_nodes, mc.NORMAL_MEAN_THRESHOLD))
    preds_a, preds_b = [], []
    for nd in node_sets:
        Xa = states_for_task(lt, nd, task_a, La); Xb = states_for_task(lt, nd, task_b, Lb)
        preds_a.append(np.vstack([Xa, np.ones((1, Xa.shape[1]))]).T @ wa)
        preds_b.append(np.vstack([Xb, np.ones((1, Xb.shape[1]))]).T @ wb)
    return dict(pred_a=np.array(preds_a), pred_b=np.array(preds_b), grid=g)


def windowed_nrmse(pred: np.ndarray, target: np.ndarray, window: int, sd: float):
    """Root-mean-square error in sliding windows (centered), divided by a fixed standard deviation."""
    err2 = (pred - target) ** 2
    kernel = np.ones(window) / window
    ms = np.convolve(err2, kernel, mode="same")
    return np.sqrt(ms) / sd


# ----------------------------------------------------------------------------- compromise objective
def compromise_objective(params: dict, tasks, refs: dict, mode: str, N_R: int, R: int, seed: int):
    """max over tasks of NRMSE_t / ref_t with the same physical parameters, L, and per-task lambda selection."""
    rows = {}
    for t in tasks:
        rows[t] = mc.evaluate(t, params, mode=mode, N_R=N_R, R=R, seed=seed)
    ratios = {t: rows[t]["objective"] / refs[t] for t in tasks}
    return max(ratios.values()), rows, ratios
