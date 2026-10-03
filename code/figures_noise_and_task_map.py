"""Figures of the noise analysis and the task map: Figure 5 of the main article (validation against the particle
simulations, occupancy traces, receptor-number sweep to 1e9 with the designed readout), Figure 7 (task map: error ratio
against receptor number per task class, critical receptor numbers, operating points), and Figures S12 and S13 of the
Supporting Information (readout grid), drawn from the result files in data/results.
Usage: python figures_noise_and_task_map.py [fig5_noise fig7_task_map figS12_S13_readout_grid]
Each figure is a function of the same name and is written to manuscript/figures/<name>.pdf (figS12_S13_readout_grid
writes figS12_readout_grid_a.pdf and figS13_readout_grid_b.pdf)."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import simulator as mc
OUT = mc.ROOT / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
WANT = sys.argv[1:] or ["fig5_noise", "fig7_task_map", "figS12_S13_readout_grid"]
BENCH_DIR = mc.RESULTS / "benchmark_tasks"; GROUPS_DIR = mc.RESULTS / "case_study_minimax_wide_domain"; ADD_DIR = mc.RESULTS / "additional_tasks_and_readouts"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False, "axes.titlesize": 9.5})
TASKS3 = ["MG", "SINE", "MGCUBED"]
TNAME = {"MG": "Mackey-Glass forecasting", "SINE": "Sine-to-square transformation", "MGCUBED": "MG-Cubed prediction", "XOR": "Temporal XOR", "PAT": "Bit-pattern detection", "NARMA": "NARMA-10", "GLUF": "Biomarker forecast", "GLUE": "Threshold decision"}
TSHORT = {"MG": "MG forecast", "SINE": "Sine-to-square", "MGCUBED": "MG-Cubed", "XOR": "XOR", "PAT": "pattern", "NARMA": "NARMA-10", "GLUF": "biomarker forecast", "GLUE": "threshold decision"}
TCOL = {"MG": "#3B75AF", "SINE": "#C8553D", "MGCUBED": "#4F9D5B", "XOR": "#8E5FB5", "PAT": "#E8862B", "NARMA": "#2A9D8F", "GLUF": "#1F4E79", "GLUE": "#B23A48"}
SCOL = {"S1": "#3B75AF", "S2": "#E8862B", "S3": "#4F9D5B", "S4": "#8E5FB5", "S0": "black"}
CLASS_ORDER = [("memory", ["MG", "NARMA", "GLUF"]), ("mixed", ["MGCUBED", "PAT"]), ("nonlinearity", ["SINE", "XOR", "GLUE"])]


def rd(p):
    p = Path(p); return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if p.exists() else []
def fl(x):
    try: return float(x)
    except Exception: return float("nan")
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)
def panel_label(ax, s, x=-0.16, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


LONG = rd(ADD_DIR / "task_map.csv"); BUDGET = {r["task"]: r for r in rd(ADD_DIR / "critical_receptor_numbers.csv")}


def series(task, s, proto):
    pts = sorted([(int(fl(r["N_R"])), fl(r["objective"])) for r in LONG if r["task"] == task and r["set"] == s and r["protocol"] == proto and int(fl(r["N_R"])) > 0])
    return [p[0] for p in pts], [p[1] for p in pts]
def meanfield(task, s):
    v = [fl(r["objective"]) for r in LONG if r["task"] == task and r["set"] == s and r["protocol"] == "meanfield"]; return v[0] if v else float("nan")
def linref(task): return fl(BUDGET[task]["linear_ref"]) if task in BUDGET else float("nan")


# ----------------------------------------------------------------------------- Figure 5 (noise)
def fig5_noise():
    pv = rd(BENCH_DIR / "particle_validation.csv")
    fig = plt.figure(figsize=(10.2, 6.6)); gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    for r in pv:
        for kind, mk in (("raw", "o"), ("filt", "s")):
            ax.errorbar(fl(r[f"smoldyn_{kind}_mean"]), fl(r[f"hybrid_{kind}_mean"]), xerr=fl(r[f"smoldyn_{kind}_sd"]), yerr=fl(r[f"hybrid_{kind}_sd"]), fmt=mk, color=TCOL[r["task"]], ms=5,
                        mfc=TCOL[r["task"]] if kind == "filt" else "white", mec=TCOL[r["task"]], capsize=1.5, elinewidth=0.7, mew=0.9)
    lim = [0.2, 1.45]; ax.plot(lim, lim, color="0.5", lw=0.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("particle model (Smoldyn), NRMSE"); ax.set_ylabel("stochastic receptor model, NRMSE"); ax.set_title("Validation, twelve configurations")
    h = [Line2D([], [], marker="o", color=TCOL[t], ls="none", ms=5, label=TSHORT[t]) for t in TASKS3] + [Line2D([], [], marker="o", color="0.4", mfc="white", ls="none", ms=5, label="unfiltered"), Line2D([], [], marker="s", color="0.4", ls="none", ms=5, label="filtered")]
    ax.legend(handles=h, fontsize=8, loc="lower right", handletextpad=0.3, labelspacing=0.3); panel_label(ax, "(a)", x=-0.3, y=1.07)
    ax = fig.add_subplot(gs[0, 1:])
    o = [r for r in rd(BENCH_DIR / "optima.csv") if r["mode"] == "det" and r["set"] == "S1" and r["task"] == "MG"][0]; p = mc.params_from_row(o); td = mc.get_task("MG")
    g, c, counts = mc.concentration(td, p); time = np.arange(g["n_time"]) * g["dt"]; b = mc.meanfield_full(c, g["dt"], p["k_on"], p["k_off"])
    n0, n1 = 1200, 1206; sl = slice(n0 * g["steps_per_symbol"], n1 * g["steps_per_symbol"])
    for NR, col, lab in ((500, "#C8553D", "$N_R=500$"), (50000, "#3B75AF", "$N_R=50000$")):
        nb = mc.markov_full(c[: n1 * g["steps_per_symbol"] + 1], g["dt"], p["k_on"], p["k_off"], NR, 1, 11)[0] / NR
        ax.plot(time[sl] - time[sl][0], nb[sl], color=col, lw=0.6, alpha=0.9, label=lab)
    ax.plot(time[sl] - time[sl][0], b[sl], color="black", lw=1.3, label="mean field")
    for k in range(n1 - n0): ax.axvline(k * p["T"], color="0.85", lw=0.6)
    ax.set_xlabel("time within six consecutive symbols (s)"); ax.set_ylabel("occupancy $b(t)$"); ax.set_title("Receptor-count noise, configuration tuned for forecasting (S1)")
    ax.legend(fontsize=8, loc="upper right", ncol=3); panel_label(ax, "(b)", x=-0.1)
    for j, task in enumerate(TASKS3):
        ax = fig.add_subplot(gs[1, j])
        for s, col, ls, lab in (("S0", "black", "--", "S0 nominal, retrained sampled readout"), ("S1", SCOL["S1"], "-", "S1 optimum, retrained sampled readout"), ("S4", SCOL["S4"], "-", "S4 optimum, retrained sampled readout")):
            xs, ys = series(task, s, "retrained"); ax.plot(xs, ys, color=col, ls=ls, lw=1.3, marker="o", ms=3.2, label=lab)
        xs, ys = series(task, "S1", "remedy")
        if xs: ax.plot(xs, ys, color=SCOL["S1"], ls="-", lw=1.3, marker="s", ms=4, mfc="white", label="S1 optimum, designed readout (grid or default)")
        xs, ys = series(task, "S4", "remedy")
        if xs: ax.plot(xs, ys, color=SCOL["S4"], ls="-", lw=1.3, marker="s", ms=4, mfc="white", label="S4, same")
        for s, mk in (("S1", "^"), ("S4", "P")):
            xs, ys = series(task, s, "noiseaware"); ax.plot(xs, ys, mk, color=SCOL[s], ms=8, mec="black", mew=0.6, ls="none", label=f"{s}, noise-aware channel", zorder=5)
        ax.axhline(meanfield(task, "S4"), color=SCOL["S4"], lw=0.8, ls="-.", label="S4 optimum, mean field")
        ax.axhline(linref(task), color="0.45", lw=0.9, ls="--", label="linear regression on the transmitted input")
        ax.set_xscale("log"); ax.set_xlabel("receptor number $N_R$"); ax.set_title(TNAME[task]); ax.set_ylim(0, 1.0); ax.set_xlim(3e2, 2e9)
        ax.axvspan(1e3, 1e5, color="0.94", lw=0); ax.axvspan(1e7, 2e9, color="0.94", lw=0)
        ax.text(1e4, 0.95, "cell range", fontsize=7.5, ha="center", color="0.35"); ax.text(1.4e8, 0.95, "electrode range", fontsize=7.5, ha="center", color="0.35")
        ax.grid(color="0.92", lw=0.6, which="both"); ax.set_axisbelow(True)
        if j == 0: ax.set_ylabel("NRMSE, optimization block")
        panel_label(ax, ["(c)", "(d)", "(e)"][j], x=-0.22 if j == 0 else -0.12)
        if j == 2: handles, labels = ax.get_legend_handles_labels()
    # column-wise groups: physical configuration with retrained readout | readout designed for noise | noise-aware channels and references
    order = ["S0 nominal, retrained sampled readout", "S1 optimum, retrained sampled readout", "S4 optimum, retrained sampled readout",
             "S1 optimum, designed readout (grid or default)", "S4, same", "S4 optimum, mean field",
             "S1, noise-aware channel", "S4, noise-aware channel", "linear regression on the transmitted input"]
    lookup = dict(zip(labels, handles)); handles = [lookup[l] for l in order if l in lookup]; labels = [l for l in order if l in lookup]
    labels = [l.replace("S4, same", "S4 optimum, designed readout (grid or default)").replace("linear regression on the transmitted input", "linear regression on the input") for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.4, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.0, labelspacing=0.4)
    fig.subplots_adjust(bottom=0.175, top=0.95); save(fig, "fig5_noise")


# ----------------------------------------------------------------------------- Figure 7 (task map)
def fig7_task_map():
    fig = plt.figure(figsize=(10.2, 7.9)); gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.3, bottom=0.2, top=0.95)
    for j, (cls, tasks) in enumerate(CLASS_ORDER):
        ax = fig.add_subplot(gs[0, j])
        for t in tasks:
            L = linref(t)
            xs, ys = series(t, "S4", "retrained"); ax.plot(xs, [y / L for y in ys], color=TCOL[t], lw=1.3, marker="o", ms=3.2, label=f"{TSHORT[t]}")
            xs, ys = series(t, "S4", "remedy")
            if xs: ax.plot(xs, [y / L for y in ys], color=TCOL[t], lw=1.2, ls="--", marker="s", ms=3.5, mfc="white")
            ax.axhline(meanfield(t, "S4") / L, color=TCOL[t], lw=0.7, ls=":")
        ax.axhline(1.0, color="0.3", lw=0.9); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e2, 2e9); ax.set_ylim({"memory": 0.3, "mixed": 0.05, "nonlinearity": 0.005}[cls], 3.0)
        ax.set_xlabel("receptor number $N_R$"); ax.set_title(f"{cls} tasks", fontsize=9.5); ax.grid(color="0.92", lw=0.6, which="both"); ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower left")
        if j == 0: ax.set_ylabel("NRMSE relative to the linear reference"); panel_label(ax, "(a)", x=-0.2)
    # (b) receptor budgets as a dot plot: receptor number needed to beat the linear reference and to reach NRMSE 0.25
    ax = fig.add_subplot(gs[1, :2])
    order = [t for cls, ts in CLASS_ORDER for t in ts]
    y = np.arange(len(order))[::-1]
    NOTREACHED = 4e9
    def val(t, key):
        v = BUDGET.get(t, {}).get(key, "none"); return NOTREACHED if v in ("none", "", None) else max(fl(v), 500.0)
    specs = (("budget_retrained_S4", "o", "white", "below the reference, default readout"), ("budget_remedy_S4", "o", None, "below the reference, designed readout"),
             ("level_retrained_S4", "s", "white", "reaches NRMSE 0.25, default readout"), ("level_remedy_S4", "s", None, "reaches NRMSE 0.25, designed readout"))
    for k, (key, mk, face, lab) in enumerate(specs):
        for yi, t in zip(y, order):
            v = val(t, key); off = (k - 1.5) * 0.18
            ax.plot(v, yi + off, mk, color=TCOL[t], mfc=face if face else TCOL[t], mec=TCOL[t], ms=7, mew=1.2, clip_on=False)
            if v >= NOTREACHED: ax.annotate("", xy=(6e9, yi + off), xytext=(3.5e9, yi + off), arrowprops=dict(arrowstyle="->", color=TCOL[t], lw=1.0))
    for yi, t in zip(y, order): ax.axhline(yi, color="0.93", lw=0.6, zorder=0)
    ax.set_xscale("log"); ax.set_xlim(3e2, 6e9); ax.set_yticks(y); ax.set_yticklabels([TSHORT[t] for t in order], fontsize=8); ax.set_ylim(-0.6, len(order) - 0.4)
    from matplotlib.transforms import blended_transform_factory
    tra = blended_transform_factory(ax.transData, ax.transAxes)   # data x, axes y: labels sit above the axes box
    ax.axvspan(1e3, 1e5, color="0.94", lw=0, zorder=0); ax.text(1e4, 1.03, "cell range", fontsize=7.5, color="0.35", ha="center", va="bottom", transform=tra); ax.axvspan(1e7, 6e9, color="0.94", lw=0, zorder=0); ax.text(2.4e8, 1.03, "electrode range", fontsize=7.5, color="0.35", ha="center", va="bottom", transform=tra)
    ax.axvline(5e5, color="0.6", lw=0.8, ls=":"); ax.text(5.5e5, 1.03, "designed readout\nevaluated to here", fontsize=7.2, color="0.45", ha="left", va="bottom", transform=tra)
    ax.set_xlabel("receptor number at the crossing (S4 optimum)")
    h = [Line2D([], [], marker="o", color="0.4", mfc="white", ls="none", ms=6, label=specs[0][3]), Line2D([], [], marker="o", color="0.4", ls="none", ms=6, label=specs[1][3]),
         Line2D([], [], marker="s", color="0.4", mfc="white", ls="none", ms=6, label=specs[2][3]), Line2D([], [], marker="s", color="0.4", ls="none", ms=6, label=specs[3][3])]
    ax.legend(handles=h, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1, handletextpad=0.4, labelspacing=0.35); ax.grid(axis="x", color="0.92", lw=0.6, which="both"); panel_label(ax, "(b)", x=-0.2)
    # (c) operating regimes of the S1 optima
    ax = fig.add_subplot(gs[1, 2])
    opt = rd(BENCH_DIR / "optima.csv") + rd(GROUPS_DIR / "optima.csv") + rd(ADD_DIR / "optima.csv")
    for cls, tasks in CLASS_ORDER:
        for t in tasks:
            o = [r for r in opt if r["mode"] == "det" and r["set"] == "S1" and r["task"] == t]
            if o: ax.plot(fl(o[0]["koff_T"]), fl(o[0]["c_peak_over_KD"]), {"memory": "o", "mixed": "D", "nonlinearity": "s"}[cls], color=TCOL[t], ms=7, mec="black", mew=0.5, label=TSHORT[t])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$k_{\\mathrm{off}}T$"); ax.set_ylabel("$c_{\\mathrm{peak}}/K_D$"); ax.set_title("S1 optima, mean field", fontsize=9)
    ax.axhline(1, color="0.9", lw=0.7); ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.3, labelspacing=0.3, columnspacing=0.8); ax.grid(color="0.94", lw=0.6, which="both"); panel_label(ax, "(c)", x=-0.25)
    save(fig, "fig7_task_map")


# ----------------------------------------------------------------------------- Figures S12 and S13 (readout grid)
def figS12_S13_readout_grid():
    """Readout grid of the S1 optima, two figures of two tasks each; every task row has its own legend row below it."""
    rem = rd(ADD_DIR / "readout_grid.csv")
    NRS = (500, 5000, 50000, 500000)
    for name, tasks in (("figS12_readout_grid_a", ("MG", "MGCUBED")), ("figS13_readout_grid_b", ("NARMA", "SINE"))):
        tasks = [t for t in tasks if any(r["task"] == t for r in rem)]
        if not tasks: continue
        fig = plt.figure(figsize=(10.2, 4.1 * len(tasks)))
        gs = gridspec.GridSpec(2 * len(tasks), 4, figure=fig, height_ratios=[1.0, 0.3] * len(tasks), hspace=0.5, wspace=0.3, left=0.06, right=0.985, top=0.95, bottom=0.05)
        for i, t in enumerate(tasks):
            handles = labels = None
            for j, NR in enumerate(NRS):
                ax = fig.add_subplot(gs[2 * i, j]); rr = [r for r in rem if r["task"] == t and r["set"] == "S1" and int(fl(r["N_R"])) == NR]
                base = [r for r in rr if r["variant"] == "sampled_default"]
                if base: ax.axhline(fl(base[0]["objective"]), color="black", lw=0.9, ls="--", label="sampled default readout")
                ax.axhline(linref(t), color="0.45", lw=0.8, ls=":", label="linear reference")
                Mps = sorted(set(int(fl(r["Mp"])) for r in rr if r["variant"] == "integrating"), reverse=True)
                for Mp, col in zip(Mps, ("#3B75AF", "#4F9D5B", "#E8862B", "#C8553D")):
                    for L, ls in ((5, ":"), (10, "--"), (20, "-")):
                        pts = sorted([(fl(r["tau_over_T"]), fl(r["objective"])) for r in rr if r["variant"] == "integrating" and int(fl(r["Mp"])) == Mp and int(fl(r["L"])) == L])
                        if pts: ax.plot([max(p[0], 0.03) for p in pts], [p[1] for p in pts], color=col, ls=ls, lw=1.0, marker="o", ms=2.5, label=f"$M'={Mp}$, $L={L}$")
                ax.set_xscale("log"); ax.set_xlim(0.025, 6); ax.set_title(f"{TSHORT[t]}, S1, $N_R={NR}$", fontsize=9)
                ax.set_xlabel("low-pass time constant $\\tau_f/T$", fontsize=8.5); ax.tick_params(labelsize=8)
                if j == 0: ax.set_ylabel("NRMSE"); handles, labels = ax.get_legend_handles_labels()
                ax.grid(color="0.93", lw=0.6)
            lax = fig.add_subplot(gs[2 * i + 1, :]); lax.axis("off")
            lax.legend(handles, labels, loc="center", ncol=6, fontsize=8.8, handletextpad=0.4, columnspacing=1.0, labelspacing=0.3, handlelength=2.2)
        save(fig, name)


if __name__ == "__main__":
    for name in WANT: globals()[name]()
