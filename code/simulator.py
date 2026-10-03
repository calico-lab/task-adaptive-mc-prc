"""Simulator of the MC channel as a reservoir: mean-field model and stochastic receptor model (noise-aware design).

Model (Section 2 of the manuscript)
  transmitter   point source, N_n = N_min + u_n (N_max - N_min) molecules at t = nT
  channel       unbounded 3D diffusion, c_R(t) = sum_n N_n h(t - nT), h(t) = (4 pi D t)^(-3/2) exp(-d^2/(4 D t)),
                contributions older than round(100 tau_D / T) symbols dropped (tau_D = d^2 / 6D), at least one symbol
  receiver      N_R identical receptors, k_on c (1 - b) - k_off b (mean field) or, in the noise-aware mode,
                a two-state Markov chain per receptor with the exact transition probabilities over one grid
                step for piecewise-constant concentration (the mean of the chain is the mean-field solution)
  readout       M virtual nodes per symbol at t_n + i T / M (i = 0 .. M-1); the state vector of symbol n
                concatenates the nodes of the L most recent symbols (M L features); ridge regression with a
                penalized intercept fitted on the training block; NRMSE with the sample standard deviation
                on the optimization block (design objective) and on the held-out block (check)
  ridge         the ridge parameter is chosen from LAMBDAS by the optimization-block error, in both modes

Time grid      dt = T / (M n_sub) with the smallest n_sub giving dt <= 1 ms, so every virtual node lies on a
               grid point for every T (no rounding of node instants). The mean-field recursion uses the exact
               exponential update for piecewise-constant c and is therefore insensitive to dt except through
               the piecewise-constant approximation of c(t).
Node variants  "sample"  node value = occupancy at the node instant (manuscript default)
               "integrate"  node value = mean occupancy over the node window (t_i, t_i + T / M]

Public functions
  evaluate(task, params, mode, N_R, R, seed, ...)   full evaluation, returns a flat dict of results
  input_tap_baseline(task, taps)                    ridge regression on the raw input history, no channel
  concentration(td, params)                         grid, c_R(t), molecule counts
  meanfield_full(c, dt, k_on, k_off)                full mean-field occupancy trace (figures)
  descriptors(td, params, g, c, occ_nodes, N_R)     dimensionless groups and occupancy statistics

The result files in data/ were computed with one thread per process (OMP_NUM_THREADS=1 and the corresponding variables
of the other numerical libraries, as the launch_*.py scripts set them). With multithreaded linear algebra, a rerun
reproduces the stored values up to round-off in the last digits (relative differences of about 1e-14).
"""
from __future__ import annotations
import math, time as clock
from pathlib import Path
import numpy as np
from numba import njit
from scipy.signal import oaconvolve

from task_data import get_task as _get_task, TaskData  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]   # the repository folder that contains code/ and data/
DATA = ROOT / "data" / "task_inputs"
RESULTS = ROOT / "data" / "results"
TASKS = ("MG", "SINE", "MGCUBED")
PARAM_NAMES = ("k_on", "k_off", "T", "distance", "N_max", "D")
NOMINAL = dict(k_on=1e-18, k_off=1.0, T=1.25, distance=7.07e-6, N_max=10100.0, D=3.16e-11)
N_MIN = 100.0
N_R_DEFAULT = 500
DT_MAX = 1.0e-3
HISTORY_FACTOR = 100.0
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
N_AVOGADRO = 6.02214076e23
SHELL_VOLUME_M3 = 4.0 / 3.0 * math.pi * ((3.2e-6) ** 3 - (3.0e-6) ** 3)
NORMAL_MEAN_THRESHOLD = 25.0   # binomial mean above which the normal approximation is drawn (avoids the slow large-count sampler)
EXACT_BINOMIAL = False         # set True to force the exact binomial sampler everywhere (robustness check)
EXACT_THRESHOLD = 1e300

DOMAIN = dict(k_on=("log", 5e-20, 2e-17), k_off=("log", 0.1, 10.0), T=("lin", 0.5, 2.0),
              N_max=("lin", 200.0, 20000.0), distance=("log", 6e-6, 50e-6), D=("log", 5e-12, 2e-10))
SETS = dict(S0=(), S1=("k_on", "k_off"), S2=("T", "N_max"), S3=("k_on", "k_off", "T", "N_max"),
            S4=("k_on", "k_off", "T", "N_max", "distance", "D"))


def get_task(task: str) -> TaskData:
    return _get_task(task, DATA)


def time_grid(td: TaskData, T: float, dt_max: float = DT_MAX) -> dict:
    M = td.base_nodes
    n_sub = max(int(math.ceil(T / (M * dt_max) - 1e-9)), 1)
    dt = T / (M * n_sub)
    sps = M * n_sub
    return dict(dt=dt, n_sub=n_sub, steps_per_symbol=sps, n_time=td.num_tot_points * sps + 1, M=M)


def encode(td: TaskData, N_max: float, input_series: np.ndarray | None = None) -> np.ndarray:
    s = td.input_series if input_series is None else np.asarray(input_series, dtype=np.float64)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return N_MIN + s * (N_max - N_MIN)


def history_symbols(T: float, distance: float, D: float, factor: float = HISTORY_FACTOR) -> int:
    tau_D = distance * distance / (6.0 * D)
    return max(int(math.floor(factor * tau_D / T + 0.5)), 1)


def impulse_response(t: np.ndarray, distance: float, D: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        h = np.power(4.0 * math.pi * D * t, -1.5) * np.exp(-(distance ** 2) / (4.0 * D * t))
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


def concentration(td: TaskData, params: dict, input_series: np.ndarray | None = None,
                  history: int | None = None, dt_max: float = DT_MAX):
    """Grid dict, concentration at the receiver on the grid, molecule counts per symbol."""
    g = time_grid(td, params["T"], dt_max)
    dt, sps, n_time = g["dt"], g["steps_per_symbol"], g["n_time"]
    counts = encode(td, params["N_max"], input_series)
    nh = history_symbols(params["T"], params["distance"], params["D"]) if history is None else int(history)
    n_kernel = min(nh * sps, n_time - 1)
    kernel = np.concatenate(([0.0], impulse_response(np.arange(1, n_kernel + 1) * dt, params["distance"], params["D"])))
    spikes = np.zeros(n_time)
    spikes[np.arange(counts.size) * sps] = counts
    c = oaconvolve(spikes, kernel)[:n_time]
    np.maximum(c, 0.0, out=c)
    return g, c, counts


@njit(cache=True)
def _meanfield_nodes(c, dt, k_on, k_off, n_sub, n_nodes):
    n_time = c.size
    samp = np.zeros(n_nodes)
    integ = np.zeros(n_nodes)
    x = 0.0
    acc = 0.0
    for t in range(1, n_time):
        r = k_on * c[t - 1] + k_off
        beq = k_on * c[t - 1] / r
        x = beq + (x - beq) * math.exp(-r * dt)
        acc += x
        if t % n_sub == 0:
            j = t // n_sub
            if j < n_nodes:
                samp[j] = x
            integ[j - 1] = acc / n_sub
            acc = 0.0
    return samp, integ


@njit(cache=True)
def meanfield_full(c, dt, k_on, k_off):
    n_time = c.size
    b = np.zeros(n_time)
    x = 0.0
    for t in range(1, n_time):
        r = k_on * c[t - 1] + k_off
        beq = k_on * c[t - 1] / r
        x = beq + (x - beq) * math.exp(-r * dt)
        b[t] = x
    return b


@njit(inline="always")
def _binom(n, p, thr):
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    m = n * p
    if m >= thr:
        y = m + math.sqrt(m * (1.0 - p)) * np.random.standard_normal() + 0.5
        if y < 0.0:
            return 0
        x = int(y)
        if x > n:
            return n
        return x
    return np.random.binomial(n, p)


@njit(cache=True)
def _markov_nodes(c, dt, k_on, k_off, NR, R, seed, n_sub, n_nodes, thr):
    np.random.seed(seed)
    n_time = c.size
    samp = np.zeros((R, n_nodes))
    integ = np.zeros((R, n_nodes))
    state = np.zeros(R, dtype=np.int64)
    acc = np.zeros(R)
    for t in range(1, n_time):
        r = k_on * c[t - 1] + k_off
        beq = k_on * c[t - 1] / r
        q = 1.0 - math.exp(-r * dt)
        p_fb = beq * q
        p_bf = (1.0 - beq) * q
        for i in range(R):
            s = state[i]
            s = s + _binom(NR - s, p_fb, thr) - _binom(s, p_bf, thr)
            state[i] = s
            acc[i] += s
        if t % n_sub == 0:
            j = t // n_sub
            for i in range(R):
                if j < n_nodes:
                    samp[i, j] = state[i] / NR
                integ[i, j - 1] = acc[i] / (n_sub * NR)
                acc[i] = 0.0
    return samp, integ


@njit(cache=True)
def markov_full(c, dt, k_on, k_off, NR, R, seed, thr=NORMAL_MEAN_THRESHOLD):
    """Full bound-count traces (R, n_time) for figures and for the particle-model comparison."""
    np.random.seed(seed)
    n_time = c.size
    nb = np.zeros((R, n_time), dtype=np.int32)
    state = np.zeros(R, dtype=np.int64)
    for t in range(1, n_time):
        r = k_on * c[t - 1] + k_off
        beq = k_on * c[t - 1] / r
        q = 1.0 - math.exp(-r * dt)
        p_fb = beq * q
        p_bf = (1.0 - beq) * q
        for i in range(R):
            s = state[i]
            s = s + _binom(NR - s, p_fb, thr) - _binom(s, p_bf, thr)
            state[i] = s
            nb[i, t] = s
    return nb


def states_from_nodes(td: TaskData, nodes: np.ndarray, L: int) -> np.ndarray:
    """(M L) by N_symbols state matrix; column n holds the nodes of symbols n-L+1 .. n (zero before symbol L-1)."""
    M, npts = td.base_nodes, td.num_tot_points
    X = np.zeros((M * L, npts))
    W = np.lib.stride_tricks.sliding_window_view(nodes, M * L)[::M]
    X[:, L - 1:] = W[: npts - L + 1].T
    return X


def block_slices(td: TaskData) -> dict:
    train = np.arange(td.washout1, td.washout1 + td.num_train_points)
    opt = np.arange(td.wheretostarttest, td.wheretostarttest + td.num_test_points)
    held = np.arange(td.washout1 + td.num_train_points, td.wheretostarttest)
    return dict(train=train, opt=opt, held=held)


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)) / np.std(target, ddof=1))


def fit_scores(td: TaskData, X: np.ndarray, lams=LAMBDAS, target: np.ndarray | None = None) -> dict:
    """Ridge readout (penalized intercept) for every lambda in lams; NRMSE per block. Returns {lam: {train, opt, held}}."""
    y = td.target_series if target is None else target
    bl = block_slices(td)
    Xb = {k: np.vstack([X[:, idx], np.ones((1, idx.size))]) for k, idx in bl.items()}
    Xtr, ytr = Xb["train"], y[bl["train"]]
    G = Xtr @ Xtr.T
    evals, V = np.linalg.eigh(G)
    evals = np.maximum(evals, 0.0)
    VtXy = V.T @ (Xtr @ ytr)
    out = {}
    for lam in lams:
        w = V @ (VtXy / (evals + lam))
        out[lam] = {k: nrmse(Xb[k].T @ w, y[bl[k]]) for k in bl}
    return out


def predict(td: TaskData, X: np.ndarray, lam: float, block: str = "opt") -> dict:
    bl = block_slices(td)
    Xtr = np.vstack([X[:, bl["train"]], np.ones((1, bl["train"].size))])
    w = np.linalg.solve(Xtr @ Xtr.T + lam * np.eye(Xtr.shape[0]), Xtr @ td.target_series[bl["train"]])
    Xb = np.vstack([X[:, bl[block]], np.ones((1, bl[block].size))])
    return dict(prediction=Xb.T @ w, target=td.target_series[bl[block]], rows=bl[block])


def descriptors(td: TaskData, params: dict, g: dict, c: np.ndarray, occ_nodes: np.ndarray, N_R: int) -> dict:
    """Dimensionless groups; concentration and mean-field occupancy statistics over the optimization block."""
    T, D, d, k_on, k_off = params["T"], params["D"], params["distance"], params["k_on"], params["k_off"]
    bl = block_slices(td)
    sps, M = g["steps_per_symbol"], g["M"]
    cw = c[bl["opt"][0] * sps:(bl["opt"][-1] + 1) * sps]
    bw = occ_nodes[bl["opt"][0] * M:(bl["opt"][-1] + 1) * M]
    KD = k_off / k_on
    tau_D = d * d / (6.0 * D)
    c_peak, c_mean = float(c.max()), float(cw.mean())
    # within-symbol contrast: mean over optimization-block symbols of (max - min) of the M node values
    per_symbol = bw.reshape(-1, M)
    contrast_within = float(np.mean(per_symbol.max(axis=1) - per_symbol.min(axis=1)))
    contrast_between = float(np.std(per_symbol.mean(axis=1)))
    binom_sd = float(np.mean(np.sqrt(np.clip(bw * (1 - bw), 0, None) / N_R)))
    return dict(
        KD_nM=KD / N_AVOGADRO * 1e6, tau_D_over_T=tau_D / T, phi_ISI=math.erf(d / math.sqrt(4.0 * D * T)),
        koff_T=k_off * T, c_peak=c_peak, c_mean=c_mean, c_peak_over_KD=c_peak / KD, c_mean_over_KD=c_mean / KD,
        tauR_mean_over_T=1.0 / (k_on * c_mean + k_off) / T, tauR_peak_over_T=1.0 / (k_on * c_peak + k_off) / T,
        occ_mean=float(bw.mean()), occ_std=float(bw.std()), occ_max=float(bw.max()),
        occ_frac_above_0p8=float(np.mean(bw > 0.8)), occ_frac_above_0p9=float(np.mean(bw > 0.9)),
        contrast_within=contrast_within, contrast_between=contrast_between, binomial_sd_mean=binom_sd,
        snr_within=contrast_within / binom_sd if binom_sd > 0 else float("inf"),
        shell_molecules_peak=c_peak * SHELL_VOLUME_M3, shell_molecules_mean=c_mean * SHELL_VOLUME_M3,
        history_symbols=history_symbols(T, d, D), dt=g["dt"], n_sub=g["n_sub"],
    )


def evaluate(task: str, params: dict, mode: str = "det", N_R: int = N_R_DEFAULT, R: int = 10, seed: int = 0,
             lams=LAMBDAS, node: str = "sample", input_series: np.ndarray | None = None,
             history: int | None = None, dt_max: float = DT_MAX, return_realizations: bool = False) -> dict:
    """Evaluate one configuration. params: k_on, k_off, T, distance, N_max, D, L.

    mode "det": mean-field occupancy (one realization). mode "hyb": R realizations of the receptor
    Markov chain with N_R receptors, each with its own readout fit; means and sds over realizations.
    The ridge parameter is the element of lams with the smallest mean optimization-block NRMSE.
    """
    t0 = clock.time()
    td = get_task(task)
    L = int(round(params["L"]))
    g, c, counts = concentration(td, params, input_series, history, dt_max)
    n_nodes = td.num_tot_points * g["M"]
    mf_samp, mf_integ = _meanfield_nodes(c, g["dt"], params["k_on"], params["k_off"], g["n_sub"], n_nodes)
    if mode == "det":
        node_sets = [mf_samp if node == "sample" else mf_integ]
    elif mode == "hyb":
        samp, integ = _markov_nodes(c, g["dt"], params["k_on"], params["k_off"], int(N_R), int(R), int(seed), g["n_sub"], n_nodes, EXACT_THRESHOLD if EXACT_BINOMIAL else NORMAL_MEAN_THRESHOLD)
        node_sets = list(samp if node == "sample" else integ)
    else:
        raise ValueError(mode)
    per = {lam: dict(train=[], opt=[], held=[]) for lam in lams}
    for nd in node_sets:
        sc = fit_scores(td, states_from_nodes(td, nd, L), lams)
        for lam in lams:
            for k in ("train", "opt", "held"):
                per[lam][k].append(sc[lam][k])
    opt_mean = {lam: float(np.mean(per[lam]["opt"])) for lam in lams}
    lam_best = min(lams, key=lambda l: opt_mean[l] if np.isfinite(opt_mean[l]) else np.inf)
    Rr = len(node_sets)
    sd = lambda v: float(np.std(v, ddof=1)) if Rr > 1 else 0.0
    row = dict(task=task, mode=mode, node=node, N_R=int(N_R) if mode == "hyb" else 0, R=Rr, seed=int(seed) if mode == "hyb" else 0,
               k_on=params["k_on"], k_off=params["k_off"], T=params["T"], distance=params["distance"],
               N_max=params["N_max"], D=params["D"], L=L, lam_best=lam_best,
               objective=opt_mean[lam_best], opt_sd=sd(per[lam_best]["opt"]),
               held=float(np.mean(per[lam_best]["held"])), held_sd=sd(per[lam_best]["held"]),
               train=float(np.mean(per[lam_best]["train"])))
    for lam in lams:
        row[f"opt_lam_{lam:.0e}"] = opt_mean[lam]
        row[f"held_lam_{lam:.0e}"] = float(np.mean(per[lam]["held"]))
    row.update(descriptors(td, params, g, c, mf_samp, int(N_R)))
    row["seconds"] = clock.time() - t0
    if return_realizations:
        row["opt_values"] = np.array(per[lam_best]["opt"])
        row["held_values"] = np.array(per[lam_best]["held"])
        row["_nodes"] = node_sets
        row["_grid"], row["_c"], row["_counts"] = g, c, counts
    return row


def input_tap_baseline(task: str, taps: int, lams=LAMBDAS) -> dict:
    """Ridge regression on the current and the taps-1 preceding inputs (no channel), same blocks and rule."""
    td = get_task(task)
    u = td.input_series
    X = np.vstack([np.concatenate([np.zeros(k), u[: u.size - k]]) for k in range(taps)])
    sc = fit_scores(td, X, lams)
    lam_best = min(lams, key=lambda l: sc[l]["opt"])
    return dict(task=task, taps=taps, lam_best=lam_best, objective=sc[lam_best]["opt"], held=sc[lam_best]["held"],
                train=sc[lam_best]["train"], **{f"opt_lam_{lam:.0e}": sc[lam]["opt"] for lam in lams})


def params_from_row(row: dict) -> dict:
    p = {k: float(row[k]) for k in PARAM_NAMES}
    p["L"] = int(round(float(row["L"] if "L" in row else row["memorywindowlength"])))
    return p


def evaluate_all_L(task: str, params: dict, mode: str = "det", N_R: int = N_R_DEFAULT, R: int = 10, seed: int = 0,
                   lams=LAMBDAS, node: str = "sample", Ls=(1, 2, 3, 4, 5), input_series: np.ndarray | None = None,
                   history: int | None = None, dt_max: float = DT_MAX) -> dict:
    """Like evaluate, but the occupancy (or the R chain realizations) is simulated once and the readout is
    fitted for every L in Ls. Returns {L: row}; every row carries the same descriptors."""
    t0 = clock.time()
    td = get_task(task)
    g, c, counts = concentration(td, params, input_series, history, dt_max)
    n_nodes = td.num_tot_points * g["M"]
    mf_samp, mf_integ = _meanfield_nodes(c, g["dt"], params["k_on"], params["k_off"], g["n_sub"], n_nodes)
    if mode == "det":
        node_sets = [mf_samp if node == "sample" else mf_integ]
    else:
        samp, integ = _markov_nodes(c, g["dt"], params["k_on"], params["k_off"], int(N_R), int(R), int(seed), g["n_sub"], n_nodes, EXACT_THRESHOLD if EXACT_BINOMIAL else NORMAL_MEAN_THRESHOLD)
        node_sets = list(samp if node == "sample" else integ)
    des = descriptors(td, params, g, c, mf_samp, int(N_R))
    out = {}
    Rr = len(node_sets)
    sd = lambda v: float(np.std(v, ddof=1)) if Rr > 1 else 0.0
    for L in Ls:
        per = {lam: dict(train=[], opt=[], held=[]) for lam in lams}
        for nd in node_sets:
            sc = fit_scores(td, states_from_nodes(td, nd, int(L)), lams)
            for lam in lams:
                for k in ("train", "opt", "held"):
                    per[lam][k].append(sc[lam][k])
        opt_mean = {lam: float(np.mean(per[lam]["opt"])) for lam in lams}
        lam_best = min(lams, key=lambda l: opt_mean[l] if np.isfinite(opt_mean[l]) else np.inf)
        row = dict(task=task, mode=mode, node=node, N_R=int(N_R) if mode == "hyb" else 0, R=Rr, seed=int(seed) if mode == "hyb" else 0,
                   k_on=params["k_on"], k_off=params["k_off"], T=params["T"], distance=params["distance"],
                   N_max=params["N_max"], D=params["D"], L=int(L), lam_best=lam_best,
                   objective=opt_mean[lam_best], opt_sd=sd(per[lam_best]["opt"]),
                   held=float(np.mean(per[lam_best]["held"])), held_sd=sd(per[lam_best]["held"]),
                   train=float(np.mean(per[lam_best]["train"])))
        for lam in lams:
            row[f"opt_lam_{lam:.0e}"] = opt_mean[lam]
            row[f"held_lam_{lam:.0e}"] = float(np.mean(per[lam]["held"]))
        row.update(des)
        row["seconds"] = clock.time() - t0
        out[int(L)] = row
    return out


def best_over_L(rows: dict) -> dict:
    """The row with the smallest objective among the rows of evaluate_all_L."""
    return min(rows.values(), key=lambda r: r["objective"] if np.isfinite(r["objective"]) else np.inf)
