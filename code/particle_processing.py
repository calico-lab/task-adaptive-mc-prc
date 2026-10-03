"""State construction and scoring with the protocol of the particle simulations (used by particle_validation.py).

The particle simulations (Smoldyn) record the receptor occupancy on a 10 ms grid, and their errors were computed with
virtual nodes at rounded indices of that grid, ridge regression with a penalized intercept, and the NRMSE with the
sample standard deviation. The functions below apply the same rules to any occupancy trace, so that the stochastic
receptor model of simulator.py can be scored exactly as the particle simulations were.

Functions
  make_cfg(row, L=None)               settings dict from a result row with k_on, k_off, T, distance, N_max, D
  build_states(td, cfg, sim, L)       M*L by N state matrix for readout length L, from sim["time"] and sim["occupation"]
  score(td, states, lam=1e-6)         NRMSE on the optimization block, the held-out block, and the training block
  block_slices(td)                    row ranges of the three blocks (0-based)
"""
from __future__ import annotations
import numpy as np

from task_data import matlab_round, TaskData  # noqa: F401

N_R = 500
LAMBDA = 1e-6


def make_cfg(row: dict, L: int | None = None) -> dict:
    Lval = int(round(float(row["memorywindowlength"]))) if L is None else int(L)
    return dict(N=N_R, N_min=100.0, lam=LAMBDA, dt=0.001, memlengthsweep=100,
                k_on=float(row["k_on"]), k_off=float(row["k_off"]), T=float(row["T"]),
                distance=float(row["distance"]), memorywindowlength=Lval,
                N_max=float(row["N_max"]), D=float(row["D"]))


def build_states(td: TaskData, cfg: dict, sim: dict, L: int, num_points: int | None = None,
                 indexing: str | None = None) -> np.ndarray:
    time, occupation = sim["time"], sim["occupation"]
    n_time = time.size
    T, dt = cfg["T"], cfg["dt"]
    steps_per_symbol = T / dt
    M = td.base_nodes
    Nres = M * L
    npts = td.num_tot_points if num_points is None else num_points
    rule = td.state_indexing if indexing is None else indexing
    states = np.zeros((Nres, npts), dtype=np.float64)
    offsets = np.arange(Nres, dtype=np.float64) * (steps_per_symbol / Nres) * L
    for symbol_index in range(npts):
        symbol_start = symbol_index * T
        if rule == "find_ge":
            pos = int(np.searchsorted(time, symbol_start, side="left"))
            if pos >= n_time:
                continue
            start_index = pos + 1
        else:
            start_index = int(matlab_round(np.array(symbol_start / dt))) + 1
        sample_idx = matlab_round(start_index - (L - 1) * steps_per_symbol + offsets).astype(np.int64)
        if np.all(sample_idx > 0) and np.all(sample_idx <= n_time):
            states[:, symbol_index] = occupation[sample_idx - 1]
    return states


def block_slices(td: TaskData) -> dict:
    train = np.arange(td.washout1, td.washout1 + td.num_train_points)
    opt = np.arange(td.wheretostarttest, td.wheretostarttest + td.num_test_points)
    held = np.arange(td.washout1 + td.num_train_points, td.wheretostarttest)
    return dict(train=train, opt=opt, held=held)


def ridge_fit(x_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    gram = x_train @ x_train.T + lam * np.eye(x_train.shape[0])
    return np.linalg.pinv(gram) @ (x_train @ y_train.reshape(-1, 1))


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)) / np.std(target, ddof=1))


def score(td: TaskData, states: np.ndarray, lam: float = LAMBDA, target: np.ndarray | None = None) -> dict:
    """Readout fitted on the training block, scored on the optimization, held-out, and training blocks."""
    y = td.target_series if target is None else target
    bl = block_slices(td)
    xs_tr, y_tr = states[:, bl["train"]], y[bl["train"]]
    if td.remove_zero_columns:
        vt = np.any(xs_tr != 0, axis=0)
        xs_tr, y_tr = xs_tr[:, vt], y_tr[vt]
    x_tr = np.vstack([xs_tr, np.ones((1, xs_tr.shape[1]))])
    w = ridge_fit(x_tr, y_tr, lam)
    out = dict(train=nrmse((x_tr.T @ w).reshape(-1), y_tr))
    for name in ("opt", "held"):
        xs, yy = states[:, bl[name]], y[bl[name]]
        if td.remove_zero_columns:
            vs = np.any(xs != 0, axis=0)
            xs, yy = xs[:, vs], yy[vs]
        x = np.vstack([xs, np.ones((1, xs.shape[1]))])
        out[name] = nrmse((x.T @ w).reshape(-1), yy)
    return out
