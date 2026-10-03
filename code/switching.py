"""Task switching in operation (manuscript subsection "Case Study and Online Task Switching").

One long run of a single S1 channel (nominal transmitter and channel, receptor rates of the source-task optimum). At
symbol n_switch the receptor rates ramp (log-linear) to the target-task optimum over `ramp` symbols and the readout is
swapped to the target-task readout, which was trained beforehand on a standalone run of the target configuration.
Three schedules are compared with R realizations of the stochastic receptor model at N_R receptors:
  switch   source configuration, then ramp to the target configuration
  B        target configuration throughout (reference level of the target task)
  A        source configuration throughout with a readout retrained for the target task on the source channel (reuse)
The error is the windowed NRMSE (WINDOW = 25 symbols, centered) against the target of the task active at that time,
divided by the standard deviation of that target over the whole record. Adaptation time = first window centre after the
switch at which the switched error stays within 10 percent of the reference level (at least 0.02) of the reference curve
for the next HOLD = 100 symbols.
The configuration of a task is the noise-aware S1 row at N_R with the lowest objective in
data/results/benchmark_tasks/optima.csv and data/results/case_study_minimax_wide_domain/optima.csv together. For MG
and MG-Cubed at 50000 receptors, this is the optimum of the wide-domain search (group wide).
Usage: python switching.py <N_R> <R> [pairs]   (the manuscript used N_R = 50000 and R = 50)
Output: data/results/case_study_minimax_wide_domain/switching (one .npz file per directed pair and a summary .json)
"""
import csv, json, sys, time
import numpy as np
import simulator_extensions as m
mc = m.mc

OPT_BENCH = mc.RESULTS / "benchmark_tasks" / "optima.csv"; OPT_GROUPS = mc.RESULTS / "case_study_minimax_wide_domain" / "optima.csv"
OUT = mc.RESULTS / "case_study_minimax_wide_domain" / "switching"; NR = int(sys.argv[1]); R = int(sys.argv[2])
PAIRS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["MG>MGCUBED", "MGCUBED>MG", "GLUF>GLUE", "GLUE>GLUF", "SINE>MG", "MG>SINE"]
RAMPS = (0, 10, 50); WINDOW = 25; N_SWITCH = 1700; HOLD = 100
OUT.mkdir(parents=True, exist_ok=True)


def optimum(task, set_="S1"):
    rows = []
    for p in (OPT_BENCH, OPT_GROUPS):
        if p.exists():
            rows += [r for r in csv.DictReader(open(p, encoding="utf-8-sig")) if r["set"] == set_ and r["task"] == task and r["mode"] == "hyb" and int(float(r["N_R"])) == NR]
    if not rows:
        raise SystemExit(f"no noise-aware S1 optimum for {task} at N_R={NR}")
    r = min(rows, key=lambda r: float(r["objective"]))
    p = mc.params_from_row(r); return p, float(r["lam_best"])


results = []
for pair in PAIRS:
    ta, tb = pair.split(">")
    pa, lama = optimum(ta); pb, lamb = optimum(tb)
    lt = m.LongTask(ta, tb, N_SWITCH * 2)
    ya, yb = lt.targets[ta], lt.targets[tb]
    sd_a, sd_b = lt.target_sd[ta], lt.target_sd[tb]
    t0 = time.time()
    # readouts: source-task readout on its own channel, target-task readout on its own channel, target-task readout retrained on the source channel
    wa, La = m.train_readout(ta, pa, NR, 501, lama)
    wb, Lb = m.train_readout(tb, pb, NR, 502, lamb)
    pa_for_b = dict(pa); pa_for_b["L"] = pb["L"]
    wb_on_a, Lb_on_a = m.train_readout(tb, pa_for_b, NR, 503, lamb)
    runs = {}
    for ramp in RAMPS:
        runs[f"switch_ramp{ramp}"] = m.switching_run(lt, ta, tb, pa, pb, wa, La, wb, Lb, N_SWITCH, ramp, NR, R, 600)
    runs["B"] = m.switching_run(lt, ta, tb, pa, pb, wa, La, wb, Lb, N_SWITCH, 0, NR, R, 600, schedule="B")
    runs["A_reuse"] = m.switching_run(lt, ta, tb, pa, pb, wa, La, wb_on_a, Lb_on_a, N_SWITCH, 0, NR, R, 600, schedule="A")
    curves = {}
    for name, run in runs.items():
        ea = np.array([m.windowed_nrmse(p, ya, WINDOW, sd_a) for p in run["pred_a"]])
        eb = np.array([m.windowed_nrmse(p, yb, WINDOW, sd_b) for p in run["pred_b"]])
        curves[name] = dict(err_a_mean=ea.mean(0), err_a_sd=ea.std(0, ddof=1) if R > 1 else 0 * ea[0], err_b_mean=eb.mean(0), err_b_sd=eb.std(0, ddof=1) if R > 1 else 0 * eb[0])
    ref = curves["B"]["err_b_mean"]
    tail = slice(N_SWITCH + 300, 2 * N_SWITCH - WINDOW)
    summary = dict(pair=pair, source=ta, target=tb, N_R=NR, R=R, n_switch=N_SWITCH, window=WINDOW, T=pa["T"],
                   source_params={k: pa[k] for k in ("k_on", "k_off", "L")}, target_params={k: pb[k] for k in ("k_on", "k_off", "L")}, lam_a=lama, lam_b=lamb,
                   level_a_before=float(curves["switch_ramp0"]["err_a_mean"][WINDOW: N_SWITCH - WINDOW].mean()),
                   level_b_reference=float(ref[tail].mean()), level_b_switched_tail=float(curves["switch_ramp0"]["err_b_mean"][tail].mean()),
                   level_b_reuse=float(curves["A_reuse"]["err_b_mean"][tail].mean()), seconds=time.time() - t0)
    for ramp in RAMPS:
        e = curves[f"switch_ramp{ramp}"]["err_b_mean"]
        peak = float(e[N_SWITCH: N_SWITCH + 200].max())
        within = np.abs(e - ref) <= max(0.1 * summary["level_b_reference"], 0.02)
        ok_from = None
        for n in range(N_SWITCH, 2 * N_SWITCH - WINDOW - HOLD):
            if within[n: n + HOLD].all():
                ok_from = n; break
        summary[f"peak_ramp{ramp}"] = peak
        summary[f"adapt_symbols_ramp{ramp}"] = (ok_from - N_SWITCH) if ok_from is not None else None
        summary[f"adapt_seconds_ramp{ramp}"] = ((ok_from - N_SWITCH) * pa["T"]) if ok_from is not None else None
    results.append(summary)
    np.savez(OUT / f"switching_{ta}_to_{tb}_NR{NR}.npz", **{f"{k}__{q}": v for k, c in curves.items() for q, v in c.items()}, ya=ya, yb=yb, n_switch=N_SWITCH, window=WINDOW, T=pa["T"])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k not in ("source_params", "target_params")}), flush=True)
with open(OUT / f"switching_summary_NR{NR}.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("wrote", OUT)
