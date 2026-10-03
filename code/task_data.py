"""Task data: input series, target series, virtual-node count, and data blocks of every task.

TaskData(task, data_dir) follows the task definitions of the original MATLAB evaluator. It also keeps that evaluator's
state-indexing rule (rounded grid index for MG and MG-Cubed, first grid time at or after the symbol start for
sine-to-square) and its removal of all-zero state columns for sine-to-square; these two rules are used only by the
processing of the particle simulations (particle_processing.py). get_task caches one object per task.
The MG, sine-to-square, and MG-Cubed series are read from the MATLAB MAT-files in data/task_inputs with the small
reader below, so that no MATLAB installation is needed. The case-study tasks (GLUF, GLUE) and the additional tasks
(XOR, PAT, NARMA) are read from the NumPy files written by case_study_signal.py and additional_task_series.py.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

MI_DOUBLE = 9
MI_MATRIX = 14
MI_COMPRESSED = 15


# ----------------------------------------------------------------------------
# MATLAB MAT-file reader (numeric arrays, compressed or not)
# ----------------------------------------------------------------------------
def _elements(buffer: bytes):
    offset = 0
    size = len(buffer)
    while offset + 4 <= size:
        first = struct.unpack_from("<I", buffer, offset)[0]
        small_bytes = first >> 16
        small_type = first & 0xFFFF
        if small_bytes:
            yield small_type, buffer[offset + 4 : offset + 4 + small_bytes]
            offset += 8
            continue
        if offset + 8 > size:
            break
        data_type, byte_count = struct.unpack_from("<II", buffer, offset)
        start = offset + 8
        end = start + byte_count
        if end > size:
            raise ValueError("Truncated MATLAB data element")
        yield data_type, buffer[start:end]
        offset = end if data_type == MI_COMPRESSED else end + ((8 - byte_count % 8) % 8)


def _parse_matrix(payload: bytes):
    items = list(_elements(payload))
    if len(items) < 4:
        raise ValueError("Incomplete miMATRIX element")
    _, dimensions_raw = items[1]
    dimensions = np.frombuffer(dimensions_raw, dtype="<i4").astype(int)
    _, name_raw = items[2]
    name = name_raw.decode("utf-8")
    real_type, real_raw = items[3]
    if real_type != MI_DOUBLE:
        return None
    values = np.frombuffer(real_raw, dtype="<f8").copy()
    return name, values.reshape(tuple(dimensions), order="F")


def load_mat(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    if not raw.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"Not a MATLAB MAT-file: {path}")
    variables: dict[str, np.ndarray] = {}
    for data_type, payload in _elements(raw[128:]):
        if data_type == MI_COMPRESSED:
            for inner_type, inner_payload in _elements(zlib.decompress(payload)):
                if inner_type == MI_MATRIX:
                    parsed = _parse_matrix(inner_payload)
                    if parsed is not None:
                        variables[parsed[0]] = parsed[1]
        elif data_type == MI_MATRIX:
            parsed = _parse_matrix(payload)
            if parsed is not None:
                variables[parsed[0]] = parsed[1]
    return variables


# ----------------------------------------------------------------------------
# MATLAB rounding
# ----------------------------------------------------------------------------
def matlab_round(x: np.ndarray) -> np.ndarray:
    """MATLAB round(): half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ----------------------------------------------------------------------------
# Task data (mirrors the MATLAB switch on task)
# ----------------------------------------------------------------------------
class TaskData:
    def __init__(self, task: str, data_dir: Path):
        task = task.upper()
        self.task = task
        if task == "MG":
            self.base_nodes, self.washout1 = 50, 500
            self.num_train_points, self.wheretostarttest = 500, 1200
            self.num_test_points, self.num_tot_points = 500, 1700
            self.state_indexing, self.remove_zero_columns = "rounded", False
            horizon = 6
            series = load_mat(data_dir / "MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat")["mackey_glass_series"].reshape(-1)
            segment = series[: self.num_tot_points + horizon]
            normalized = (segment - segment.min()) / (segment.max() - segment.min())
            self.input_series = normalized[:-horizon].copy()
            self.target_series = normalized[horizon:].copy()
        elif task == "SINE":
            self.base_nodes, self.washout1 = 100, 500
            self.num_train_points, self.wheretostarttest = 2000, 3000
            self.num_test_points, self.num_tot_points = 1000, 4000
            self.state_indexing, self.remove_zero_columns = "find_ge", True
            values = load_mat(data_dir / "SINEseries_len5000_p25.mat")
            self.input_series = values["input_sine_series"].reshape(-1)[: self.num_tot_points].copy()
            self.target_series = values["target_square_series"].reshape(-1)[: self.num_tot_points].copy()
        elif task == "MGCUBED":
            self.base_nodes, self.washout1 = 50, 500
            self.num_train_points, self.wheretostarttest = 500, 1200
            self.num_test_points, self.num_tot_points = 500, 1700
            self.state_indexing, self.remove_zero_columns = "rounded", False
            values = load_mat(data_dir / "MGCubed_series_k10.mat")
            self.input_series = values["input_series"].reshape(-1)[: self.num_tot_points].copy()
            self.target_series = values["target_series"].reshape(-1)[: self.num_tot_points].copy()
        elif task in ("GLUF", "GLUE"):
            # case study (glucose-like biomarker); same blocks and node count as the MG tasks
            self.base_nodes, self.washout1 = 50, 500
            self.num_train_points, self.wheretostarttest = 500, 1200
            self.num_test_points, self.num_tot_points = 500, 1700
            self.state_indexing, self.remove_zero_columns = "rounded", False
            d = np.load(data_dir / "glucose_case_study.npz", allow_pickle=True)
            self.input_series = d["input"][: self.num_tot_points].astype(float).copy()
            self.target_series = (d["target_forecast"] if task == "GLUF" else d["target_episode"])[: self.num_tot_points].astype(float).copy()
        elif task in ("XOR", "PAT", "NARMA"):
            # additional tasks on an on-off keyed bit stream (XOR, PAT) and NARMA-10; same blocks and node count as the MG tasks
            self.base_nodes, self.washout1 = 50, 500
            self.num_train_points, self.wheretostarttest = 500, 1200
            self.num_test_points, self.num_tot_points = 500, 1700
            self.state_indexing, self.remove_zero_columns = "rounded", False
            d = np.load(data_dir / "additional_tasks.npz")
            if task == "NARMA":
                self.input_series = d["narma_input"][: self.num_tot_points].astype(float).copy()
                self.target_series = d["narma_target"][: self.num_tot_points].astype(float).copy()
            else:
                self.input_series = d["bits"][: self.num_tot_points].astype(float).copy()
                self.target_series = (d["target_xor"] if task == "XOR" else d["target_pat"])[: self.num_tot_points].astype(float).copy()
        else:
            raise ValueError(f"Unknown task {task}")


_TASK_CACHE: dict[str, TaskData] = {}


def get_task(task: str, data_dir: Path) -> TaskData:
    key = task.upper()
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = TaskData(key, data_dir)
    return _TASK_CACHE[key]
