"""Case-study signal: a glucose-like biomarker from the Bergman minimal model driven by meals.

Model (per minute). G glucose (mg/dL), X remote insulin action (1/min), I plasma insulin (mU/L).
  dG/dt = -p1 (G - Gb) - X G + Ra(t) / V_G
  dX/dt = -p2 X + p3 s(t) (I - Ib)
  dI/dt = -n (I - Ib) + sigma max(G - Gb, 0)
Meal glucose appearance (Hovorka et al. 2004): Ra(t) = D A_G t exp(-t / t_max) / t_max^2 for each meal of size D (mg/kg).
s(t) is a 24 h modulation of insulin sensitivity (dawn phenomenon), s = 1 - a cos(2 pi (t - t_min)/24 h), and the basal
glucose Gb(t) carries a slow Ornstein-Uhlenbeck fluctuation (sd 4 mg/dL, correlation time 120 min) that stands for the
unmodelled variability of hepatic glucose output.
Meals at about 07:30, 12:30, 19:00 with jitter, sizes drawn once per meal, and an optional afternoon snack.
Integration by RK4 with a 1 min step over 14 days, sampled every 5 min (4032 samples). Everything depends on one seed.

Tasks derived from the sampled series G_n (one sample per symbol of the MC channel):
  GLUF  forecast, target G_{n+6} (30 min ahead)
  GLUE  threshold decision, target 1 if G_n > 140 mg/dL (the level is above the threshold now), else 0
Input and forecast target are normalized to [0, 1] over the whole record. Output: data/task_inputs/glucose_case_study.npz.
Usage: python case_study_signal.py [seed]   (default seed 2028, the record used in the manuscript)
"""
import sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "task_inputs" / "glucose_case_study.npz"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2028
P = dict(Gb=90.0, Ib=10.0, p1=0.028, p2=0.025, p3=1.3e-5, n=0.14, sigma=0.05, VG=1.7, AG=0.8, tmax=40.0, a=0.2, tmin_h=4.0, ou_sd=4.0, ou_tau=120.0)
DAYS, DT, SAMPLE = 14, 1.0, 5
THRESH, H_F, H_E = 140.0, 6, 12


def meals(rng):
    """(time in min, size in mg/kg) for every meal of the record."""
    out = []
    for d in range(DAYS):
        base = d * 1440.0
        for hour, mean_size in ((7.5, 700.0), (12.5, 850.0), (19.0, 950.0)):
            t = base + 60.0 * (hour + rng.normal(0.0, 0.6))
            size = mean_size * math.exp(rng.normal(0.0, 0.25))
            out.append((t, size))
        if rng.random() < 0.35:
            out.append((base + 60.0 * (16.0 + rng.normal(0.0, 0.5)), 300.0 * math.exp(rng.normal(0.0, 0.3))))
    return sorted(out)


def ra(t, meal_list):
    total = 0.0
    for tm, D in meal_list:
        tau = t - tm
        if 0.0 < tau < 600.0:
            total += D * P["AG"] * tau * math.exp(-tau / P["tmax"]) / P["tmax"] ** 2
    return total


def deriv(t, y, meal_list, gb):
    G, X, I = y
    s = 1.0 - P["a"] * math.cos(2.0 * math.pi * (t / 60.0 - P["tmin_h"]) / 24.0)
    dG = -P["p1"] * (G - gb) - X * G + ra(t, meal_list) / P["VG"]
    dX = -P["p2"] * X + P["p3"] * s * (I - P["Ib"])
    dI = -P["n"] * (I - P["Ib"]) + P["sigma"] * max(G - P["Gb"], 0.0)
    return np.array([dG, dX, dI])


def simulate(seed):
    rng = np.random.default_rng(seed)
    meal_list = meals(rng)
    n_steps = int(DAYS * 1440 / DT)
    y = np.array([P["Gb"], 0.0, P["Ib"]])
    G = np.zeros(n_steps + 1); G[0] = y[0]
    ou = 0.0; theta = DT / P["ou_tau"]; ou_step = P["ou_sd"] * math.sqrt(2.0 * theta)
    for k in range(n_steps):
        t = k * DT
        ou += -theta * ou + ou_step * rng.normal()
        gb = P["Gb"] + ou
        k1 = deriv(t, y, meal_list, gb); k2 = deriv(t + DT / 2, y + DT / 2 * k1, meal_list, gb)
        k3 = deriv(t + DT / 2, y + DT / 2 * k2, meal_list, gb); k4 = deriv(t + DT, y + DT * k3, meal_list, gb)
        y = y + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        G[k + 1] = y[0]
    return G[::SAMPLE], meal_list


if __name__ == "__main__":
    G, meal_list = simulate(SEED)
    N = G.size
    episode = (G > THRESH).astype(float)
    forecast = np.roll(G, -H_F); forecast[-H_F:] = np.nan
    valid = N - H_E
    lo, hi = G[:valid].min(), G[:valid].max()
    inp = (G[:valid] - lo) / (hi - lo); tf = (forecast[:valid] - lo) / (hi - lo); te = episode[:valid]
    np.savez(OUT, glucose_mgdl=G[:valid], input=inp, target_forecast=tf, target_episode=te, threshold=THRESH, horizon_forecast=H_F, horizon_episode=H_E,
             sample_minutes=SAMPLE, seed=SEED, meal_times_min=np.array([m[0] for m in meal_list]), meal_sizes_mgkg=np.array([m[1] for m in meal_list]), params=np.array(list(P.items()), dtype=object))
    print(f"saved {OUT}: {valid} samples ({valid*SAMPLE/1440:.1f} days), glucose {lo:.0f} to {hi:.0f} mg/dL, mean {G[:valid].mean():.0f}, above {THRESH:.0f} for {100*np.mean(G[:valid] > THRESH):.0f} percent of the time, threshold target positive {100*te.mean():.0f} percent")
