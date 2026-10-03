"""Figures of the case study: Figure 8 of the main article (case study and task switching), and Figures S10 (case-study
signal) and S11 (memory and nonlinear capacity against N_R) of the Supporting Information, drawn from the result files
in data/results/case_study_minimax_wide_domain and from data/task_inputs/glucose_case_study.npz.
Usage: python figures_case_study.py [fig8_switching figS10_case_signal figS11_capacity]
Each figure is a function of the same name and is written to manuscript/figures/<name>.pdf."""
import csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import simulator as mc
OUT = mc.ROOT / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
WANT = sys.argv[1:] or ["fig8_switching", "figS10_case_signal", "figS11_capacity"]
RES = mc.RESULTS / "case_study_minimax_wide_domain"
SW = RES / "switching"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False, "axes.titlesize": 9.5})
TCOL = {"MG": "#3B75AF", "SINE": "#C8553D", "MGCUBED": "#4F9D5B", "GLUF": "#3B75AF", "GLUE": "#C8553D"}
RCOL = {0: "#C8553D", 10: "#E8862B", 50: "#8E5FB5"}
TS = {"MG": "MG forecasting", "MGCUBED": "MG-Cubed", "GLUF": "biomarker forecast", "GLUE": "threshold decision", "SINE": "sine-to-square"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)


def panel_label(ax, s, x=-0.16, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def switch_panel(ax, npz, title, show_legend):
    d = np.load(npz); n_sw = int(d["n_switch"]); T = float(d["T"]); W = int(d["window"])
    n = np.arange(d["B__err_b_mean"].size)
    x = (n - n_sw)  # symbols relative to the switch
    lo, hi = -150, 400
    sel = (x >= lo) & (x <= hi)
    ax.axvspan(lo, 0, color="0.93", lw=0)
    pre = sel & (x < 0)
    ax.plot(x[pre], d["switch_ramp0__err_a_mean"][pre], color="0.35", lw=1.0, label="source task, source configuration and readout")
    for ramp in (0, 10, 50):
        e = d[f"switch_ramp{ramp}__err_b_mean"]
        ax.plot(x[sel & (x >= 0)], e[sel & (x >= 0)], color=RCOL[ramp], lw=1.1, label=f"switched, ramp {ramp} symbols")
    ax.plot(x[sel & (x >= 0)], d["B__err_b_mean"][sel & (x >= 0)], color="black", lw=1.0, ls="--", label="target configuration throughout")
    ax.plot(x[sel & (x >= 0)], d["A_reuse__err_b_mean"][sel & (x >= 0)], color="0.5", lw=1.0, ls=":", label="source configuration kept, readout retrained")
    ax.axvline(0, color="0.3", lw=0.8)
    ymax = min(1.6, 1.15 * max(d["B__err_b_mean"][sel & (x >= 0)].max(), d["A_reuse__err_b_mean"][sel & (x >= 0)].max()))
    ax.set_xlim(lo, hi); ax.set_ylim(0, ymax)
    peak0 = d["switch_ramp0__err_b_mean"][(x >= 0) & (x < 200)].max()
    if peak0 > ymax: ax.annotate(f"peak {peak0:.1f}, above the axis", xy=(3, ymax * 0.995), xytext=(150, ymax * 0.93), textcoords="data", fontsize=8.5, color=RCOL[0], va="center", ha="left", arrowprops=dict(arrowstyle="->", color=RCOL[0], lw=0.9, shrinkA=0, shrinkB=1))
    ax.set_xlabel(f"symbols after the switch (symbol interval {T:.2f} s)"); ax.set_title(title, fontsize=9)
    ax.grid(color="0.92", lw=0.6); ax.set_axisbelow(True)
    if show_legend: ax.legend(fontsize=8, loc="upper right", handletextpad=0.4, labelspacing=0.3)


def fig8_switching():
    NR = 50000
    fig = plt.figure(figsize=(10.2, 6.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.25, bottom=0.155)
    # (a) case-study signal with the two targets, first 400 symbols of the optimization block
    ax = fig.add_subplot(gs[0, :])
    d = np.load(mc.DATA / "glucose_case_study.npz", allow_pickle=True)
    g = d["glucose_mgdl"]; thr = float(d["threshold"]); ep = d["target_episode"]
    n0, n1 = 1200, 1600; n = np.arange(n0, n1); t_h = (n - n0) * float(d["sample_minutes"]) / 60.0
    ax.plot(t_h, g[n0:n1], color="black", lw=1.0, label="biomarker (input)")
    ax.plot(t_h, np.roll(g, -6)[n0:n1], color=TCOL["GLUF"], lw=0.9, ls="--", label="forecast target (30 min ahead)")
    ax.axhline(thr, color="0.5", lw=0.7, ls=":")
    ax2 = ax.twinx(); ax2.fill_between(t_h, 0, ep[n0:n1], color=TCOL["GLUE"], alpha=0.25, lw=0, step="mid", label="threshold target (level above the illustrative 140 mg/dL)")
    ax2.set_ylim(0, 4); ax2.set_yticks([0, 1]); ax2.set_ylabel("threshold target", color=TCOL["GLUE"])
    ax.set_xlabel("time (h), optimization block of the case study"); ax.set_ylabel("glucose (mg/dL)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", ncol=3)
    ax.set_title("Synthetic glucose case study, one input and two targets", fontsize=9); panel_label(ax, "(a)", x=-0.06)
    # (b, c) switching
    ax = fig.add_subplot(gs[1, 0]); switch_panel(ax, SW / f"switching_GLUF_to_GLUE_NR{NR}.npz", f"Biomarker forecast to threshold decision, S1, $N_R={NR}$", False)
    ax.set_ylabel("windowed NRMSE"); panel_label(ax, "(b)", x=-0.14)
    hh, ll = ax.get_legend_handles_labels(); fig.legend(hh, ll, loc="lower center", ncol=3, fontsize=8.6, handletextpad=0.5, columnspacing=1.5, bbox_to_anchor=(0.5, 0.01))
    ax = fig.add_subplot(gs[1, 1]); switch_panel(ax, SW / f"switching_SINE_to_MG_NR{NR}.npz", f"Sine-to-square to MG forecasting, S1, $N_R={NR}$", False)
    panel_label(ax, "(c)", x=-0.14)
    save(fig, "fig8_switching")


def figS10_case_signal():
    d = np.load(mc.DATA / "glucose_case_study.npz", allow_pickle=True)
    g = d["glucose_mgdl"]; t_h = np.arange(g.size) * float(d["sample_minutes"]) / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(10.2, 4.6), sharex=True)
    ax[0].plot(t_h, g, color="black", lw=0.8); ax[0].axhline(float(d["threshold"]), color="#C8553D", lw=0.8, ls="--")
    for tm in d["meal_times_min"]: ax[0].plot(tm / 60.0, 72, "^", color="0.5", ms=3)
    box = dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5)   # white backing so that the labels stay legible over the peaks
    ax[0].axvspan(0, 1700 * float(d["sample_minutes"]) / 60, color="0.93", lw=0); ax[0].text(2, 205, "standard task blocks (symbols 1 to 1700)", fontsize=8, bbox=box, zorder=5)
    ax[0].axvspan(1700 * float(d["sample_minutes"]) / 60, 3400 * float(d["sample_minutes"]) / 60, color="0.97", lw=0); ax[0].text(145, 205, "second half of the switching run (symbols 1701 to 3400)", fontsize=8, bbox=box, zorder=5)
    ax[0].set_ylim(65, 235); ax[0].set_ylabel("glucose (mg/dL)")
    ax[1].plot(t_h, d["input"], color="#3B75AF", lw=0.7, label="input $u(n)$"); ax[1].plot(t_h, d["target_episode"], color="#C8553D", lw=0.7, label="threshold target"); ax[1].set_xlabel("time (h)")
    ax[1].set_ylim(-0.05, 1.32); ax[1].legend(fontsize=8, loc="upper right", ncol=2, frameon=True, framealpha=0.95, edgecolor="none", fancybox=False)
    fig.tight_layout(); save(fig, "figS10_case_signal")


def figS11_capacity():
    rows = list(csv.DictReader(open(RES / "capacity.csv")))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for ax, key, lab in ((axes[0], "linear_capacity", "linear memory capacity"), (axes[1], "quadratic_capacity", "second-order capacity")):
        for task in ("MG", "SINE", "MGCUBED"):
            sub = sorted([r for r in rows if r["config"] == f"S1_{task}"], key=lambda r: float(r["N_R"]))
            xs = [float(r["N_R"]) for r in sub]; ys = [float(r[key]) for r in sub]; es = [float(r[key + "_sd"]) for r in sub]
            ax.errorbar(xs, ys, yerr=es, color=TCOL[task], marker="o", ms=4, lw=1.1, capsize=2, label=f"S1 tuned for {TS[task]}")
            mf = [r for r in rows if r["config"] == f"S1_{task}" and r["N_R"] == "0"]
            if mf: ax.axhline(float(mf[0][key]), color=TCOL[task], lw=0.8, ls="--")
        nom = sorted([r for r in rows if r["config"] == "nominal"], key=lambda r: float(r["N_R"]))
        ax.errorbar([float(r["N_R"]) for r in nom if r["N_R"] != "0"], [float(r[key]) for r in nom if r["N_R"] != "0"], yerr=[float(r[key + "_sd"]) for r in nom if r["N_R"] != "0"], color="black", marker="x", ms=5, lw=1.0, capsize=2, label="nominal")
        mf = [r for r in nom if r["N_R"] == "0"]
        if mf: ax.axhline(float(mf[0][key]), color="black", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_xlabel("receptor number $N_R$"); ax.set_ylabel(lab); ax.grid(color="0.92", lw=0.6, which="both")
    axes[0].legend(fontsize=7); axes[0].text(0.02, 0.02, "dashed, mean field", transform=axes[0].transAxes, fontsize=7, color="0.4")
    ax = axes[2]
    for NR, col, lab in (("0", "black", "mean field"), ("500", "#C8553D", "$N_R=500$"), ("50000", "#3B75AF", "$N_R=50000$"), ("500000", "#8E5FB5", "$N_R=500000$")):
        r = [x for x in rows if x["config"] == "S1_MG" and x["N_R"] == NR][0]
        ax.plot(range(1, 11), [float(r[f"lin_k{k}"]) for k in range(1, 11)], marker="o", ms=3.5, lw=1.1, color=col, label=lab)
        ax.plot(range(0, 6), [float(r[f"quad_k{k}"]) for k in range(0, 6)], marker="s", ms=3.5, lw=1.1, ls="--", color=col)
    ax.set_xlabel("delay $k$ (symbols)"); ax.set_ylabel("$r^2$ of the delayed target"); ax.set_ylim(0, 1.05); ax.grid(color="0.92", lw=0.6)
    ax.set_title("S1 forecasting optimum, $u(n-k)$ (solid), $P_2$ (dashed)", fontsize=8); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "figS11_capacity")


if __name__ == "__main__":
    for name in WANT:
        globals()[name]()
