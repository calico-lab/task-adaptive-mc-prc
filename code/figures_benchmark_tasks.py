"""Figures of the benchmark tasks: Figures 1 to 4 and 6 of the main article, and Figures S1 to S3 and S7 to S9 of the
Supporting Information. Every panel is drawn from the result files in data/results/benchmark_tasks or from the model
equations through simulator.py; no value is typed in.

Usage: python figures_benchmark_tasks.py [figure ...]
Each figure is a function of the same name and is written to manuscript/figures/<name>.pdf; without arguments, all
figures of this script are drawn: fig1_system fig2_error_surfaces fig3_operating_points fig4_tunable_sets
fig6_noise_aware figS1_channel_error_surface figS2_reuse figS3_reuse_heldout figS7_convergence_noise_aware
figS8_error_surfaces_noise figS9_receptor_number_sweep
"""
import csv, sys, math, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib import gridspec
import matplotlib.patheffects as pe
import simulator as mc

OUT = mc.ROOT / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
WANT = sys.argv[1:] or ["fig1_system", "fig2_error_surfaces", "fig3_operating_points", "fig4_tunable_sets", "fig6_noise_aware",
                        "figS1_channel_error_surface", "figS2_reuse", "figS3_reuse_heldout", "figS7_convergence_noise_aware",
                        "figS8_error_surfaces_noise", "figS9_receptor_number_sweep"]
RES = mc.RESULTS / "benchmark_tasks"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                     "legend.frameon": False, "axes.titlesize": 9.5})
TASKS = ["MG", "SINE", "MGCUBED"]
TNAME = {"MG": "Mackey-Glass forecasting", "SINE": "Sine-to-square transformation", "MGCUBED": "MG-Cubed prediction"}
TSHORT = {"MG": "MG", "SINE": "Sine-to-square", "MGCUBED": "MG-Cubed"}
TCOL = {"MG": "#3B75AF", "SINE": "#C8553D", "MGCUBED": "#4F9D5B"}
SCOL = {"S1": "#3B75AF", "S2": "#E8862B", "S3": "#4F9D5B", "S4": "#8E5FB5", "S0": "black"}
SMARK = {"S1": "^", "S2": "s", "S3": "D", "S4": "P", "S0": "x"}
NRCOL = {500: "#C8553D", 5000: "#E8862B", 50000: "#3B75AF", 500000: "#8E5FB5"}
SETS = ["S1", "S2", "S3", "S4"]


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)


def panel_label(ax, s, x=-0.16, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


SEARCHES = rd(RES / "searches.csv"); OPTIMA = rd(RES / "optima.csv"); S0 = rd(RES / "s0.csv")


def det_searches(s, task):
    return [r for r in SEARCHES if r["mode"] == "det" and r["set"] == s and r["task"] == task]


def s0_det(task):
    rows = [r for r in S0 if r["mode"] == "det" and r["task"] == task]
    return min(fl(r["objective_mean"]) for r in rows)


def optimum(mode, NR, s, task):
    for r in OPTIMA:
        if r["mode"] == mode and int(r["N_R"]) == NR and r["set"] == s and r["task"] == task:
            return r
    return None


# ----------------------------------------------------------------------------- Figure 1
def fig1_system():
    from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
    td = mc.get_task("MG")
    seg = td.input_series[40:52].copy(); n_show = seg.size
    p = dict(mc.NOMINAL, L=3)
    # replay the short segment alone through the model equations
    class _TD:  # minimal task-like object for the schematic replay
        pass
    tds = _TD(); tds.base_nodes, tds.num_tot_points, tds.input_series, tds.task = td.base_nodes, n_show, seg, "MG"
    g, c, Ntx = mc.concentration(tds, p, input_series=seg)
    time = np.arange(g["n_time"]) * g["dt"]
    b = mc.meanfield_full(c, g["dt"], p["k_on"], p["k_off"])
    nb = mc.markov_full(c, g["dt"], p["k_on"], p["k_off"], 500, 1, 3)[0] / 500.0
    T, D, d = p["T"], p["D"], p["distance"]

    # layout in inches from the top edge: (a) the generic MC channel above its signal chain, (b) the two dynamical
    # resources, (c) the representative MC systems; every part is sized to its content, so that no empty band remains
    W, H = 10.2, 8.62
    fig = plt.figure(figsize=(W, H))
    def rect(top, height, left, width):
        return [left, 1.0 - (top + height) / H, width, height / H]
    L0, R0, GAP = 0.075, 0.985, 0.062   # horizontal extent of parts a and b, gap between the signal-chain panels
    w4 = (R0 - L0 - 3 * GAP) / 4
    SCH_TOP, SCH_H, A_TOP, A_H = 0.08, 1.25, 1.40, 1.70
    B_TOP, B_H, C_TOP = 3.98, 1.62, 6.45
    WC, GC = 0.1856, 0.012               # width of a system sketch and the gap between sketches (figure fractions)
    C_H = WC * W * 10.3 / 10.0           # sketch height at the aspect of its drawing (x from 0 to 10, y from -2.6 to 7.7)

    # ---- helpers that draw the components (shared by the channel in a and the systems in c)
    def ycap(ax, x, y, ang, size, color, lw=0.9):
        a = math.radians(ang); tx, ty = x + size * math.cos(a), y + size * math.sin(a)
        ax.plot([x, tx], [y, ty], color=color, lw=lw, solid_capstyle="round", zorder=6)
        for da in (38, -38):
            aa = a + math.radians(da); ax.plot([tx, tx + 0.62 * size * math.cos(aa)], [ty, ty + 0.62 * size * math.sin(aa)], color=color, lw=lw, solid_capstyle="round", zorder=6)
    def cell(ax, x, y, r, face, edge, nucleus=True, lw=0.9):
        ax.add_patch(Circle((x, y), r, fc=face, ec=edge, lw=lw, zorder=3))
        if nucleus: ax.add_patch(Circle((x - 0.22 * r, y + 0.12 * r), 0.36 * r, fc=edge, ec="none", alpha=0.3, zorder=4))
    def cell_receptors(ax, x, y, r, color, n=7, start=110, stop=250, size=0.5):
        for ang in np.linspace(start, stop, n):
            a = math.radians(ang); ycap(ax, x + r * math.cos(a), y + r * math.sin(a), ang, size, color)
    def molecules(ax, x0, x1, y, n, seed, color="0.3", spread0=0.25, spread1=1.25, ms=2.0):
        rng = np.random.default_rng(seed)
        xs = x0 + (x1 - x0) * rng.random(n) ** 0.85
        frac = (xs - x0) / (x1 - x0)
        ys = y + rng.normal(0, 1, n) * (spread0 + (spread1 - spread0) * frac)
        for xx, yy, fr in zip(xs, ys, frac): ax.plot(xx, yy, "o", color=color, ms=ms, alpha=float(0.9 - 0.55 * fr), mec="none", zorder=2)
    def readout(ax, x, y, color, from_xy=None):
        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.45), 2.6, 0.9, boxstyle="round,pad=0.08", fc="white", ec=color, lw=1.0, zorder=7))
        ax.text(x, y, "readout", ha="center", va="center", fontsize=7.0, color=color, zorder=8)
        if from_xy: ax.annotate("", xy=(x, y - 0.53), xytext=from_xy, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7), zorder=6)
    def electrode(ax, x, y, w, h, face, edge, rec_color, n=5, size=0.45):
        ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=face, ec=edge, lw=0.9, zorder=3))
        for yy in np.linspace(y - 0.36 * h, y + 0.36 * h, n): ycap(ax, x, yy, 180, size, rec_color)
    def label(ax, x, y, s, color="0.35", fs=6.8, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center"); ax.text(x, y, s, fontsize=fs, color=color, linespacing=1.05, zorder=9, **kw)
    def dots(ax, tun, color):
        names = ["$T$, $N_{\\max}$", "$k_{\\mathrm{on}}$, $k_{\\mathrm{off}}$", "$d$, $D$", "$L$, $\\lambda$"]
        for xx, nm, t in zip((1.5, 3.85, 6.2, 8.55), names, tun):
            ax.plot(xx, -1.0, "o", ms=6.5, mfc=color if t else "white", mec="0.35", mew=0.8, zorder=5, clip_on=False)
            ax.text(xx, -1.95, nm, fontsize=6.8, ha="center", va="center", color="0.25")
    def sketch_S0(ax):
        col = "0.25"
        ax.plot(1.4, 3.6, "o", ms=8, color="0.45", zorder=3); label(ax, 1.4, 2.5, "point\ntransmitter")
        molecules(ax, 1.9, 6.7, 3.6, 46, 1); label(ax, 4.2, 5.9, "diffusing\nmolecules")
        cell(ax, 7.6, 3.6, 1.15, "0.93", "0.5", nucleus=False); cell_receptors(ax, 7.6, 3.6, 1.15, "0.45")
        label(ax, 7.7, 1.4, "receiver with\nligand receptors")
        readout(ax, 7.6, 6.8, col, from_xy=(7.6, 4.78))
    def sketch_S1(ax):
        col = SCOL["S1"]
        ax.add_patch(FancyBboxPatch((0.35, 0.85), 9.3, 5.75, boxstyle="round,pad=0.1", fc="#F8ECE7", ec="#E2CCC3", lw=0.8, zorder=0))
        label(ax, 1.5, 6.15, "tissue", color="#A5776A", fs=6.8)
        for (xx, yy) in ((1.25, 3.05), (2.25, 3.95), (1.45, 4.55)): cell(ax, xx, yy, 0.52, "0.94", "0.55")
        label(ax, 1.75, 1.75, "physiological\nsource")
        molecules(ax, 3.0, 6.9, 3.6, 46, 2)
        electrode(ax, 7.35, 3.75, 1.65, 3.0, "0.88", "0.45", col, n=5)
        label(ax, 8.2, 1.45, "implanted\nbiosensor")
        readout(ax, 7.6, 7.15, col, from_xy=(8.15, 5.25))
    def sketch_S2(ax):
        col = SCOL["S2"]
        ax.add_patch(FancyBboxPatch((0.5, 2.7), 1.8, 1.9, boxstyle="round,pad=0.08", fc="#FCE7D3", ec=col, lw=1.0, zorder=3))
        ax.add_patch(Rectangle((2.3, 3.4), 0.55, 0.45, fc=col, ec=col, lw=0.8, zorder=3))
        label(ax, 1.4, 5.35, "engineered\nsource", color=col)
        label(ax, 1.5, 1.75, "programmed\nrelease", color=col)
        molecules(ax, 3.05, 6.7, 3.6, 46, 3)
        cell(ax, 7.6, 3.6, 1.15, "0.93", "0.5"); cell_receptors(ax, 7.6, 3.6, 1.15, "0.45")
        label(ax, 7.7, 1.35, "native cell,\nnative receptors")
        readout(ax, 7.6, 6.8, col, from_xy=(7.6, 4.78))
    def sketch_S3(ax):
        col = SCOL["S3"]
        cell(ax, 1.75, 3.6, 1.1, "#E4F1E5", col); ax.add_patch(Circle((1.6, 3.35), 0.3, fc="none", ec=col, lw=0.9, zorder=5))
        label(ax, 1.75, 1.5, "engineered\nsender cell", color=col)
        molecules(ax, 3.0, 6.5, 3.6, 40, 4)
        cell(ax, 7.6, 3.6, 1.1, "#E4F1E5", col); cell_receptors(ax, 7.6, 3.6, 1.1, col)
        label(ax, 7.7, 1.5, "engineered\nreceiver cell", color=col)
        ax.annotate("", xy=(2.95, 5.45), xytext=(6.3, 5.45), arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8, mutation_scale=7)); label(ax, 4.45, 6.0, "medium and $d$ imposed", fs=6.6)
        readout(ax, 7.6, 6.85, col, from_xy=(7.6, 4.73))
    def sketch_S4(ax):
        col = SCOL["S4"]
        ax.add_patch(FancyBboxPatch((0.4, 0.95), 9.2, 5.35, boxstyle="round,pad=0.1", fc="#F4EFF9", ec="0.6", lw=0.9, zorder=0))
        ax.add_patch(Rectangle((1.15, 2.95), 7.7, 1.3, fc="white", ec="0.55", lw=0.8, zorder=1))
        label(ax, 0.9, 5.95, "microfluidic platform", color="0.4", fs=6.8, ha="left")
        ax.plot(1.9, 3.6, "o", ms=8, color=col, zorder=3); label(ax, 1.95, 1.95, "programmed\ntransmitter", color=col)
        molecules(ax, 2.4, 6.75, 3.6, 40, 5, spread0=0.15, spread1=0.55)
        electrode(ax, 7.35, 3.6, 1.25, 2.3, "#E8DEF1", col, col, n=5, size=0.4)
        label(ax, 7.75, 1.65, "receiver, chosen\nreceptors", color=col)
        ax.annotate("", xy=(2.3, 4.7), xytext=(7.15, 4.7), arrowprops=dict(arrowstyle="<->", color=col, lw=0.8, mutation_scale=7)); label(ax, 4.7, 5.15, "$d$, $D$ adjustable", color=col, fs=6.6)
        readout(ax, 7.6, 7.0, col, from_xy=(8.0, 4.78))

    # ---- (a) the generic MC channel, each component drawn above the panel of its signal
    sch = fig.add_axes(rect(SCH_TOP, SCH_H, L0, R0 - L0)); sch.axis("off")
    sch.set_xlim(0, (R0 - L0) * W); sch.set_ylim(0, SCH_H)          # data units are inches, so circles stay round
    cx = [(i * (w4 + GAP) + w4 / 2) * W for i in range(4)]          # centers of the four signal-chain panels
    yc, rr = 0.90, 0.29                                              # height of the channel axis, receiver radius
    tx, rx = cx[0] + 0.30, cx[2]
    sch.plot(tx, yc, "o", ms=9, color="0.45", zorder=4)
    molecules(sch, tx + 0.12, rx - rr - 0.12, yc, 62, 7, spread0=0.02, spread1=0.11, ms=2.3)
    cell(sch, rx, yc, rr, "0.93", "0.5", nucleus=False)
    cell_receptors(sch, rx, yc, rr, "0.45", n=7, size=0.12)
    for ang in (147, 180, 213):                                      # a few bound receptors
        a_ = math.radians(ang); sch.plot(rx + (rr + 0.165) * math.cos(a_), yc + (rr + 0.165) * math.sin(a_), "o", color="0.25", ms=2.6, zorder=7)
    sch.annotate("", xy=(tx, yc - 0.31), xytext=(rx, yc - 0.31), arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8, mutation_scale=7))
    sch.text(cx[1], yc - 0.34, "distance $d$, diffusion coefficient $D$", ha="center", va="top", fontsize=7.2, color="0.35")
    bx0, bx1 = cx[3] - 0.62, cx[3] + 0.62
    sch.add_patch(FancyBboxPatch((bx0, yc - 0.19), bx1 - bx0, 0.38, boxstyle="round,pad=0.03", fc="white", ec="0.25", lw=1.0, zorder=7))
    sch.text(cx[3], yc, "readout", ha="center", va="center", fontsize=8, color="0.25", zorder=8)
    sch.annotate("", xy=(bx0 - 0.03, yc), xytext=(rx + rr + 0.06, yc), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.9, mutation_scale=8))
    sch.text((rx + rr + bx0) / 2, yc + 0.05, "bound receptors", ha="center", va="bottom", fontsize=7.2, color="0.35")
    sch.annotate("", xy=(bx1 + 0.28, yc), xytext=(bx1 + 0.03, yc), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.9, mutation_scale=8))
    sch.text(bx1 + 0.16, yc + 0.07, "$\\hat{y}(n)$", ha="center", va="bottom", fontsize=8, color="0.25")
    for x, (nm, pr) in zip(cx, (("Transmitter", "$T$, $N_{\\max}$"), ("Diffusion channel", "$d$, $D$"),
                                ("Ligand-receptor receiver", "$k_{\\mathrm{on}}$, $k_{\\mathrm{off}}$, $N_R$"), ("Linear readout", "$M$, $L$, $\\lambda$"))):
        sch.text(x, 0.215, nm, ha="center", va="bottom", fontsize=9)
        sch.text(x, 0.02, pr, ha="center", va="bottom", fontsize=9)
    for i in range(3):
        sch.text((cx[i] + cx[i + 1]) / 2, 0.27, "$\\rightarrow$", ha="center", va="center", fontsize=15)
    axs = [fig.add_axes(rect(A_TOP, A_H, L0 + i * (w4 + GAP), w4)) for i in range(4)]
    ax = axs[0]
    n = np.arange(1, n_show + 1)
    ax.stem(n, Ntx, linefmt="0.35", markerfmt="o", basefmt=" ")
    ax.set_xlabel("symbol $n$"); ax.set_ylabel("released molecules $N_{\\mathrm{tx}}(n)$", fontsize=8)
    ax.set_xlim(0.3, n_show + 0.7); ax.set_ylim(0, 1.15 * Ntx.max())
    ax = axs[1]
    for k in range(n_show):
        t0 = k * T; tt = time[time > t0] - t0
        ax.plot(time[time > t0], Ntx[k] * mc.impulse_response(tt, d, D) / 1e18, color=TCOL["MG"], lw=0.5, alpha=0.45)
    ax.plot(time, c / 1e18, color="black", lw=1.2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("$c_R(t)$ ($10^{18}$ m$^{-3}$)", fontsize=8)
    ax.set_xlim(0, n_show * T)
    ax = axs[2]
    ax.plot(time, nb, color="0.6", lw=0.5, label="one realization, $N_R=500$")
    ax.plot(time, b, color="black", lw=1.2, label="mean field")
    M = 10; L = p["L"]; nsym = 7
    for j in range(L):
        s0 = (nsym - 1 - j) * T
        ax.axvspan(s0, s0 + T, color=TCOL["MGCUBED"] if j == 0 else "0.85", alpha=0.25 if j == 0 else 0.5, lw=0)
    tn = (nsym - 1) * T + np.arange(M) * T / M
    ax.plot(tn, np.interp(tn, time, b), "o", color=TCOL["SINE"], ms=3.2, zorder=4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("occupancy $b(t)$", fontsize=8); ax.set_xlim(0, n_show * T); ax.set_ylim(0, 1)
    ax.annotate("$M$ virtual nodes\nper symbol", xy=(tn[4], np.interp(tn[4], time, b)), xytext=(0.66, 0.30), textcoords="axes fraction",
                fontsize=7.2, arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"), ha="center")
    # the shaded columns are the L states the readout holds: L-1 past symbols (gray) and the current symbol (green)
    ax.text((nsym - L) * T + 0.5 * L * T, 0.93, f"readout memory, $L={L}$ states", fontsize=7.2, ha="center")
    ax.text((nsym - L) * T + 0.35 * (L - 1) * T, 0.87, f"{L - 1} past\nstates", fontsize=6.6, ha="center", va="top", color="0.3", linespacing=1.0)
    ax.text((nsym - 1) * T + 0.08 * T, 0.87, "current\nstate", fontsize=6.6, ha="left", va="top", color=TCOL["MGCUBED"], linespacing=1.0)
    ax.legend(fontsize=6.5, loc="lower right", handlelength=1.5, labelspacing=0.2)
    ax = axs[3]
    ax.axis("off")
    ax.text(0.5, 0.72, r"$\hat{y}(n)=\mathbf{W}_{\mathrm{out}}^{\mathsf{T}}\,\mathbf{z}(n)$", ha="center", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.52, r"$\mathbf{z}(n)=[1,\mathbf{x}_n^{\mathsf{T}},\ldots,\mathbf{x}_{n-L+1}^{\mathsf{T}}]^{\mathsf{T}}$", ha="center", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.5, 0.28, "ridge regression on the\ntraining block, the only\ntrained part", ha="center", fontsize=7.8, transform=ax.transAxes)
    ax.text(0.5, 0.02, "forecast, transformation,\ndecision, or pattern", ha="center", fontsize=7.8, transform=ax.transAxes, color="0.3")
    fig.text(0.012, 1 - 0.20 / H, "(a)", fontsize=10, fontweight="bold")
    fig.text(0.012, 1 - (B_TOP - 0.30) / H, "(b)", fontsize=10, fontweight="bold")
    fig.text(0.012, 1 - (C_TOP - 0.36) / H, "(c)", fontsize=10, fontweight="bold")
    wb = (R0 - L0 - 0.075) / 2
    axb = [fig.add_axes(rect(B_TOP, B_H, L0, wb)), fig.add_axes(rect(B_TOP, B_H, L0 + wb + 0.075, wb))]
    ax = axb[0]
    tt = np.linspace(1e-3, 6 * T, 1500)
    o_mg, o_sine = optimum("det", 0, "S4", "MG"), optimum("det", 0, "S4", "SINE")
    curves = [(d, D, "black", f"nominal, $\\tau_D/T={d*d/(6*D)/T:.2f}$")]
    for o, task in ((o_mg, "MG"), (o_sine, "SINE")):
        if o: curves.append((fl(o["distance"]), fl(o["D"]), TCOL[task], f"tuned for {TSHORT[task]}, $\\tau_D/T={fl(o['tau_D_over_T']):.3g}$"))
    for (dd, DD, col, lab) in curves:
        h = mc.impulse_response(tt, dd, DD); ax.plot(tt / T, h / h.max(), color=col, lw=1.1, label=lab)
    h0 = mc.impulse_response(tt, d, D)
    ax.fill_between(tt / T, 0, h0 / h0.max(), where=tt > T, color="0.8", alpha=0.6, lw=0)
    ax.axvline(1, color="0.4", lw=0.7, ls="--"); ax.text(1.55, 0.35, "$\\phi_{\\mathrm{ISI}}$: normalized area\nof the tail after $T$", fontsize=8)
    ax.set_xlabel("time after release, $t/T$"); ax.set_ylabel("channel impulse\nresponse (normalized)")
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.05); ax.legend(fontsize=7.0, loc="upper right"); ax.set_title("Diffusion memory\n(source of fading memory)", fontsize=9)
    ax = axb[1]
    x = np.logspace(-2, 2.3, 400); ax.semilogx(x, x / (1 + x), color="black", lw=1.3)
    for zi, task in enumerate(("SINE", "MGCUBED", "MG")):   # MG drawn last: its range overlaps the lower end of the MG-Cubed range
        rows = [r for s in ("S1", "S3", "S4") for r in det_searches(s, task)]
        if not rows: continue
        lo = st.median(fl(r["c_mean_over_KD"]) for r in rows); hi = st.median(fl(r["c_peak_over_KD"]) for r in rows)
        xs = np.logspace(math.log10(lo), math.log10(hi), 60)
        ax.plot(xs, xs / (1 + xs), color=TCOL[task], lw=4.5, solid_capstyle="butt", zorder=3 + zi)
        ax.annotate(TSHORT[task], xy=(hi, hi / (1 + hi)), xytext=(hi * 1.9, hi / (1 + hi) - (0.16 if task == "SINE" else 0.13)), fontsize=7.5, color=TCOL[task],
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=TCOL[task]), va="center")
    ax.text(0.03, 0.97, "colored segments, operating range of\nthe receivers tuned for each task\n(median mean-to-peak concentration)", fontsize=7.0, transform=ax.transAxes, va="top")
    ax.set_xlabel("concentration relative to affinity, $c/K_D$"); ax.set_ylabel("equilibrium occupancy $b_\\infty$")
    ax.set_ylim(0, 1.02); ax.set_title("Receptor operating point\n(source of nonlinearity)", fontsize=9)


    # ---- (c) the representative MC systems drawn as MC channels; colored parts are tunable in that system, gray parts imposed
    heads = [("S0", "nominal channel,\nreadout tuned only", "0.25"), ("S1", "biosensor in tissue", SCOL["S1"]), ("S2", "engineered source,\nnative receiver", SCOL["S2"]),
             ("S3", "engineered\ntransceiver pair", SCOL["S3"]), ("S4", "in vitro platform", SCOL["S4"])]
    tuns = [[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
    for j, (draw, (s, desc, col), tun) in enumerate(zip((sketch_S0, sketch_S1, sketch_S2, sketch_S3, sketch_S4), heads, tuns)):
        ax = fig.add_axes(rect(C_TOP, C_H, 0.02 + j * (WC + GC), WC)); ax.set_xlim(0, 10); ax.set_ylim(-2.6, 7.7); ax.set_aspect("equal"); ax.axis("off")
        draw(ax); dots(ax, tun, col)
        ax.text(0.5, 1.10, s, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)
        ax.text(0.5, 1.085, desc, transform=ax.transAxes, ha="center", va="top", fontsize=7.4, color="0.3", linespacing=1.05)
    fig.text(0.5, 1 - (C_TOP + C_H + 0.06) / H, "colored components: tunable in that system; gray components: imposed.   Dots: filled, tunable parameter group; open, imposed",
             ha="center", va="top", fontsize=7.4, color="0.3")
    save(fig, "fig1_system")


# ----------------------------------------------------------------------------- Figure 2
def landscape_panel(ax, rows, xa, ya, logx, logy, vmin, vmax, xlabel, ylabel, key="nrmse_opt_best"):
    xs = sorted(set(fl(r[xa]) for r in rows)); ys = sorted(set(fl(r[ya]) for r in rows))
    Z = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        Z[ys.index(fl(r[ya])), xs.index(fl(r[xa]))] = fl(r[key])
    X, Y = np.meshgrid(xs, ys)
    im = ax.pcolormesh(X, Y, Z, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis_r", shading="nearest", rasterized=True)
    ax.contour(X, Y, Z, levels=[vmin * f for f in (1.1, 1.25, 1.5, 2.0)], colors="white", linewidths=0.5, alpha=0.8)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return im, Z


def kd_lines(ax):
    """Contours of constant K_D, drawn white with a dark outline so that they stay visible on the whole color scale."""
    outline = [pe.Stroke(linewidth=2.2, foreground="0.15", alpha=0.7), pe.Normal()]
    for KD_nM, k_label in ((0.1, 0.7), (1, 1.8), (10, 2.0), (100, None)):   # k_off at which each label sits, clear of the search optima
        slope = KD_nM * 1e-9 * 1e3 * mc.N_AVOGADRO
        kon = np.logspace(math.log10(5e-20), math.log10(2e-17), 50); koff = slope * kon
        ax.plot(kon, koff, color="white", lw=0.8, ls=(0, (1.5, 2.0)), path_effects=outline)
        if k_label is None: continue   # the 100 nM contour only clips the upper-left corner and is left unlabeled
        kx = min(1.5e-17, k_label / slope); ky = slope * kx
        ax.text(kx * 0.97, ky, f"{KD_nM:g} nM", color="white", fontsize=7.5, ha="right", va="top" if KD_nM < 1 else "bottom", clip_on=True,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="0.15")])


def fig2_error_surfaces():
    rec = rd(RES / "error_surface_receiver.csv"); tra = rd(RES / "error_surface_transmitter.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for j, task in enumerate(TASKS):
        rows = [r for r in rec if r["task"] == task]
        vmin = np.nanmin([fl(r["nrmse_opt_best"]) for r in rows]); vmax = min(1.0, 4 * vmin)
        ax = axes[0, j]
        im, Z = landscape_panel(ax, rows, "k_on", "k_off", True, True, vmin, vmax, "$k_{\\mathrm{on}}$ (m$^3$ s$^{-1}$)", "$k_{\\mathrm{off}}$ (s$^{-1}$)" if j == 0 else "")
        kd_lines(ax)
        pts = det_searches("S1", task)
        ax.plot([fl(r["k_on"]) for r in pts], [fl(r["k_off"]) for r in pts], "o", mfc="white", mec="black", ms=4.6, mew=0.8, label="S1 searches (ten)", clip_on=False)
        ax.plot(mc.NOMINAL["k_on"], mc.NOMINAL["k_off"], "x", color="black", ms=7, mew=1.4, label="nominal")
        ax.set_xlim(5e-20, 2e-17); ax.set_ylim(0.1, 10); ax.set_title(TNAME[task])
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("NRMSE, best $L$ and $\\lambda$" if j == 2 else "", fontsize=9); cb.ax.tick_params(labelsize=8)
        rows = [r for r in tra if r["task"] == task]
        vmin = np.nanmin([fl(r["nrmse_opt_best"]) for r in rows]); vmax = min(1.0, 4 * vmin)
        ax = axes[1, j]
        im, Z = landscape_panel(ax, rows, "T", "N_max", False, False, vmin, vmax, "symbol interval $T$ (s)", "release amplitude $N_{\\max}$" if j == 0 else "")
        pts = det_searches("S2", task)
        ax.plot([fl(r["T"]) for r in pts], [fl(r["N_max"]) for r in pts], "s", mfc="white", mec="black", ms=4.6, mew=0.8, label="S2 searches (ten)", clip_on=False)
        ax.plot(mc.NOMINAL["T"], mc.NOMINAL["N_max"], "x", color="black", ms=7, mew=1.4)
        ax.set_xlim(0.5, 2.0); ax.set_ylim(200, 20000)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("NRMSE, best $L$ and $\\lambda$" if j == 2 else "", fontsize=9); cb.ax.tick_params(labelsize=8)
    axes[0, 0].legend(loc="lower left", fontsize=7, handletextpad=0.3); axes[1, 0].legend(loc="upper left", fontsize=7, handletextpad=0.3)
    panel_label(axes[0, 0], "(a)"); panel_label(axes[1, 0], "(b)")
    fig.tight_layout(w_pad=0.6, h_pad=1.2)
    save(fig, "fig2_error_surfaces")


# ----------------------------------------------------------------------------- Figure 3
def fig3_operating_points():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    ax = axes[0]
    for y, lab in [(1, "0.50"), (3, "0.75"), (9, "0.90"), (19, "0.95")]:
        ax.axhline(y, color="0.88", lw=0.7); ax.text(0.36, y * 1.05, f"$p_{{\\mathrm{{eq}}}}(c_{{\\mathrm{{peak}}}})={lab}$", fontsize=7.2, color="0.45", va="bottom")
    ax.axvline(1, color="0.88", lw=0.7); ax.text(1.03, 0.62, "$k_{\\mathrm{off}}T=1$", fontsize=6.8, color="0.45", rotation=90, va="bottom")
    for task in TASKS:
        for s, ms, alpha in [("S4", 5.5, 0.85), ("S1", 4.5, 0.8), ("S3", 4.0, 0.8), ("S2", 3.5, 0.5)]:
            pts = det_searches(s, task)
            ax.plot([fl(r["koff_T"]) for r in pts], [fl(r["c_peak_over_KD"]) for r in pts], SMARK[s], color=TCOL[task], ms=ms, alpha=alpha,
                    mfc=TCOL[task] if s != "S2" else "white", mec=TCOL[task], mew=0.8)
    nom = mc.evaluate("MG", dict(mc.NOMINAL, L=5))
    ax.plot([nom["koff_T"]], [nom["c_peak_over_KD"]], "x", color="black", ms=8, mew=1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.3, 40); ax.set_ylim(0.6, 120)
    ax.set_xlabel("scaled dissociation rate, $k_{\\mathrm{off}}T$"); ax.set_ylabel("peak concentration relative to affinity, $c_{\\mathrm{peak}}/K_D$")
    ax.set_title("Receptor operating point of the tuned configurations")
    ax = axes[1]
    for task in TASKS:
        for s, ms, alpha in [("S4", 5.5, 0.85), ("S1", 4.5, 0.8), ("S3", 4.0, 0.8), ("S2", 3.5, 0.5)]:
            pts = det_searches(s, task)
            ax.plot([fl(r["tau_D_over_T"]) for r in pts], [fl(r["occ_frac_above_0p8"]) for r in pts], SMARK[s], color=TCOL[task], ms=ms, alpha=alpha,
                    mfc=TCOL[task] if s != "S2" else "white", mec=TCOL[task], mew=0.8)
    ax.plot([nom["tau_D_over_T"]], [nom["occ_frac_above_0p8"]], "x", color="black", ms=8, mew=1.5)
    ax.set_xscale("log"); ax.set_xlim(0.02, 1.0)
    ax.set_xlabel("diffusion timescale relative to the symbol, $\\tau_D/T$"); ax.set_ylabel("fraction of time with occupancy above 0.8")
    ax.set_title("Diffusion memory against saturation dwell")
    handles = [Line2D([], [], marker="o", color=TCOL[t], ls="none", ms=6, label=TSHORT[t]) for t in TASKS]
    handles += [Line2D([], [], marker=SMARK[s], color="0.4", ls="none", ms=5, mfc="0.4" if s != "S2" else "white", label=l) for s, l in
                [("S1", "S1 receiver tunable"), ("S2", "S2 transmitter tunable"), ("S3", "S3 both ends tunable"), ("S4", "S4 all tunable")]]
    handles += [Line2D([], [], marker="x", color="black", ls="none", ms=7, mew=1.5, label="nominal configuration")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
    panel_label(axes[0], "(a)", x=-0.2, y=1.02); panel_label(axes[1], "(b)", x=-0.2, y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig3_operating_points")


# ----------------------------------------------------------------------------- Figure 4
def fig4_tunable_sets():
    run = rd(RES / "running_min.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.9))
    for j, task in enumerate(TASKS):
        ax = axes[0, j]; s0 = s0_det(task)
        ax.axhline(s0, color="black", lw=0.9, ls="--"); ax.text(3.55, s0, "S0", fontsize=7.5, va="bottom", ha="right")
        for xi, s in enumerate(SETS, 1):
            v = np.array([fl(r["objective"]) for r in det_searches(s, task)])
            ax.plot(xi + np.linspace(-0.12, 0.12, v.size), v, "o", color=SCOL[s], ms=3.2, alpha=0.55, mec="none")
            ax.errorbar(xi, v.mean(), yerr=v.std(ddof=1), fmt="o", color=SCOL[s], ms=7, mec="white", mew=0.7, capsize=3, elinewidth=1.3, zorder=4)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(SETS); ax.set_xlim(0.4, 4.6)
        ax.set_title(TNAME[task]); ax.set_xlabel("tunable-parameter set"); ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
        if j == 0: ax.set_ylabel("NRMSE, optimization block")
        ax = axes[1, j]
        for s in SETS:
            rows = sorted([r for r in run if r["mode"] == "det" and r["set"] == s and r["task"] == task], key=lambda r: int(r["eval_index"]))
            ax.plot([int(r["eval_index"]) for r in rows], [fl(r["mean_running_min"]) for r in rows], color=SCOL[s], lw=1.3, label=s)
        ax.axhline(s0, color="black", lw=0.9, ls="--")
        ax.set_xlim(1, 100); ax.set_xlabel("objective evaluations"); ax.set_yscale("log")
        if j == 0: ax.set_ylabel("mean running minimum of NRMSE (ten searches)")
        ax.grid(color="0.92", lw=0.6, which="both"); ax.set_axisbelow(True)
        if j == 2: ax.legend(fontsize=7.5, loc="upper right")
    panel_label(axes[0, 0], "(a)"); panel_label(axes[1, 0], "(b)")
    fig.tight_layout(h_pad=1.5)
    save(fig, "fig4_tunable_sets")


# ----------------------------------------------------------------------------- Figures S2 and S3 (mean-field reuse)
def reuse_rows(mode, NR=0):
    return [r for r in rd(RES / "reuse.csv") if r["mode"] == mode and int(r["N_R"]) == NR]


def reuse_figure(key, name, ylabel):
    d = defaultdict(list)
    for r in reuse_rows("det"):
        d[(r["set"], r["source_task"], r["target_task"])].append(fl(r[key]))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    SRC_MK = {"MG": "o", "SINE": "s", "MGCUBED": "^"}
    for j, target in enumerate(TASKS):
        ax = axes[j]; s0 = s0_det(target)
        ax.axhline(s0, color="black", lw=0.9, ls="--"); ax.text(4.45, s0, "S0", fontsize=7.5, va="bottom", ha="right")
        for xi, s in enumerate(SETS, 1):
            ref = np.array(d[(s, target, target)])
            ax.plot(xi, np.median(ref), "*", color=SCOL[s], ms=13, mec="black", mew=0.5, zorder=5)
            offs = [-0.2, 0.2]
            for k, src in enumerate([t for t in TASKS if t != target]):
                v = np.array(d[(s, src, target)]); q1, q2, q3 = np.percentile(v, [25, 50, 75])
                ax.errorbar(xi + offs[k], q2, yerr=[[q2 - q1], [q3 - q2]], fmt=SRC_MK[src], color=TCOL[src], ms=6, capsize=2.5, elinewidth=1.0, mec="white", mew=0.5, zorder=4)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(SETS); ax.set_xlim(0.4, 4.6)
        ax.set_title(f"Target task: {TSHORT[target]}"); ax.set_xlabel("tunable-parameter set of the searches")
        ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
        if j == 0: ax.set_ylabel(ylabel)
    handles = [Line2D([], [], marker="*", color="0.4", ls="none", ms=12, mec="black", label="tuned for the target task (median of ten searches)")]
    handles += [Line2D([], [], marker=SRC_MK[t], color=TCOL[t], ls="none", ms=6, label=f"tuned for {TSHORT[t]}, readout retrained") for t in TASKS]
    handles += [Line2D([], [], color="black", ls="--", lw=0.9, label="nominal configuration, tuned readout (S0)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, name)


def figS2_reuse():
    reuse_figure("objective", "figS2_reuse", "NRMSE, optimization block")


def figS3_reuse_heldout():
    reuse_figure("held", "figS3_reuse_heldout", "NRMSE, held-out block")


# ----------------------------------------------------------------------------- Figure S1 (channel error surface)
def figS1_channel_error_surface():
    ch = rd(RES / "error_surface_channel.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for j, task in enumerate(TASKS):
        rows = [r for r in ch if r["task"] == task]
        vmin = np.nanmin([fl(r["nrmse_opt_best"]) for r in rows]); vmax = min(1.0, 4 * vmin)
        ax = axes[j]
        im, Z = landscape_panel(ax, rows, "distance", "D", True, True, vmin, vmax, "distance $d$ ($\\mu$m)", "$D$ (m$^2$ s$^{-1}$)" if j == 0 else "")
        pts = det_searches("S4", task)
        ax.plot([fl(r["distance"]) for r in pts], [fl(r["D"]) for r in pts], "P", mfc="white", mec="black", ms=5, mew=0.7, label="S4 searches (ten)")
        ax.plot(mc.NOMINAL["distance"], mc.NOMINAL["D"], "x", color="black", ms=7, mew=1.4, label="nominal")
        ax.set_xticks([6e-6, 1e-5, 2e-5, 5e-5]); ax.set_xticklabels(["6", "10", "20", "50"]); ax.set_xticks([], minor=True)
        ax.set_title(TNAME[task])
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("NRMSE, best $L$ and $\\lambda$" if j == 2 else "", fontsize=9); cb.ax.tick_params(labelsize=8)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    save(fig, "figS1_channel_error_surface")



# ----------------------------------------------------------------------------- Figures 6 and S7 to S9 (receptor-count noise)
def reeval():
    return rd(RES / "reevaluation.csv")


def get_re(rows, dm, dnr, s, task, proto, nr, node="sample"):
    for r in rows:
        if r["design_mode"] == dm and int(fl(r["design_NR"])) == dnr and r["set"] == s and r["task"] == task and r["protocol"] == proto and int(fl(r["eval_NR"])) == nr and r["node"] == node:
            return r
    return None


def baseline_best(task):
    bt = rd(RES / "baseline_taps.csv")
    return min(fl(r["objective"]) for r in bt if r["task"] == task)


def fig6_noise_aware():
    re_ = reeval()
    fig = plt.figure(figsize=(10.2, 7.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.32, top=0.88, bottom=0.17)
    NRD = [500, 5000, 50000, 500000]
    for j, task in enumerate(TASKS):
        ax = fig.add_subplot(gs[0, j])
        for k, dnr in enumerate(NRD):
            off = (k - 1.5) * 0.2   # the four receptor numbers centered on each set
            xs, ys, es = [], [], []
            for xi, s in enumerate(["S0"] + SETS):
                if s == "S0":
                    r = get_re(re_, "det", 0, "S0", task, "retrained", dnr)
                else:
                    r = get_re(re_, "hyb", dnr, s, task, "designed", dnr)
                if r: xs.append(xi + off); ys.append(fl(r["objective"])); es.append(fl(r["opt_sd"]))
            ax.errorbar(xs, ys, yerr=es, fmt="o", color=NRCOL[dnr], ms=5.5, capsize=2, elinewidth=0.9, mec="white", mew=0.5, label=f"$N_R={dnr}$, noise-aware optimum", zorder=4)
            xs2, ys2 = [], []
            for xi, s in enumerate(["S0"] + SETS):
                if s == "S0": continue
                r = get_re(re_, "det", 0, s, task, "retrained", dnr)
                if r: xs2.append(xi + off); ys2.append(fl(r["objective"]))
            ax.plot(xs2, ys2, "_", color=NRCOL[dnr], ms=10, mew=1.6, label=f"$N_R={dnr}$, mean-field optimum retrained" if j == 0 else None)
        ax.axhline(baseline_best(task), color="0.45", lw=0.9, ls="--")
        ax.set_xticks(range(5)); ax.set_xticklabels(["S0"] + SETS); ax.set_xlim(-0.5, 4.5); ax.set_title(TNAME[task]); ax.set_xlabel("tunable-parameter set")
        ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
        if j == 0: ax.set_ylabel("NRMSE, independent realizations"); panel_label(ax, "(a)", x=-0.22)
    h = [Line2D([], [], marker="o", color=NRCOL[n], ls="none", ms=5.5, label=f"$N_R={n}$") for n in NRD]
    h += [Line2D([], [], marker="o", color="0.4", ls="none", ms=5.5, label="noise-aware optimum, independent realizations"), Line2D([], [], marker="_", color="0.4", ls="none", ms=10, mew=1.6, label="mean-field optimum, readout retrained under noise"),
          Line2D([], [], color="0.45", lw=0.9, ls="--", label="linear regression on the input")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=8.4, bbox_to_anchor=(0.5, 0.995), handletextpad=0.4, columnspacing=1.2, labelspacing=0.4)
    # (b) operating points of the noise-aware optima
    ax = fig.add_subplot(gs[1, 0])
    for y, lab in [(1, "0.50"), (3, "0.75"), (9, "0.90")]:
        ax.axhline(y, color="0.9", lw=0.7); ax.text(0.11, y * 1.06, f"$p_{{\\mathrm{{eq}}}}(c_{{\\mathrm{{peak}}}})={lab}$", fontsize=8, color="0.45", va="bottom")
    NRMK = {500: "v", 5000: "D", 50000: "^", 500000: "s"}
    for task in TASKS:
        for s in ("S1", "S4"):
            pts = []
            for dnr in NRD:
                o = optimum("hyb", dnr, s, task)
                if o: pts.append((dnr, fl(o["koff_T"]), fl(o["c_peak_over_KD"])))
            for dnr, x, y in pts:
                ax.plot(x, y, NRMK[dnr], color=TCOL[task], ms=6 if s == "S1" else 7, mfc=TCOL[task] if s == "S1" else "white", mec=TCOL[task], mew=1.0)
            if len(pts) > 1: ax.plot([q[1] for q in pts], [q[2] for q in pts], color=TCOL[task], lw=0.6, alpha=0.5, ls="-" if s == "S1" else ":")
            o = optimum("det", 0, s, task)
            ax.plot(fl(o["koff_T"]), fl(o["c_peak_over_KD"]), "*", color=TCOL[task], ms=11 if s == "S1" else 12, mfc=TCOL[task] if s == "S1" else "white", mec=TCOL[task], mew=1.0, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.08, 40); ax.set_ylim(0.02, 200)
    ax.set_xlabel("$k_{\\mathrm{off}}T$"); ax.set_ylabel("$c_{\\mathrm{peak}}/K_D$"); ax.set_title("Operating points of the optima", fontsize=8.5)
    # key to (b) as a figure-level legend below the bottom row: task colors, marker shapes for the design receptor number, fill for the set
    h = [Line2D([], [], ls="none", label="legend for (b):")] + [Line2D([], [], marker="s", color=TCOL[t], ls="none", ms=7, label=TSHORT[t]) for t in TASKS]
    h += [Line2D([], [], marker="*", color="0.4", ls="none", ms=11, label="mean-field optimum")] + [Line2D([], [], marker=NRMK[n], color="0.4", ls="none", ms=7, label=f"noise-aware, $N_R={n}$") for n in NRD]
    h += [Line2D([], [], marker="s", color="0.4", ls="none", ms=7, label="S1 (filled)"), Line2D([], [], marker="s", color="0.4", mfc="white", ls="none", ms=7, label="S4 (open)")]
    fig.legend(handles=h, fontsize=8.4, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, handletextpad=0.4, columnspacing=1.3, labelspacing=0.4); panel_label(ax, "(b)", x=-0.22)
    # (c, d) receiver landscapes under noise for forecasting at the two receptor numbers
    for k, NR in enumerate((500, 50000)):
        ax = fig.add_subplot(gs[1, 1 + k])
        path = RES / f"error_surface_receiver_hyb_NR{NR}.csv"
        if not path.exists(): ax.axis("off"); continue
        rows = [r for r in rd(path) if r["task"] == "MG"]
        vmin = np.nanmin([fl(r["nrmse_opt_best"]) for r in rows]); vmax = min(1.0, 2.5 * vmin)
        im, Z = landscape_panel(ax, rows, "k_on", "k_off", True, True, vmin, vmax, "$k_{\\mathrm{on}}$ (m$^3$ s$^{-1}$)", "$k_{\\mathrm{off}}$ (s$^{-1}$)" if k == 0 else "")
        kd_lines(ax)
        pts = [r for r in SEARCHES if r["mode"] == "hyb" and int(fl(r["N_R"])) == NR and r["set"] == "S1" and r["task"] == "MG"]
        ax.plot([fl(r["k_on"]) for r in pts], [fl(r["k_off"]) for r in pts], "o", mfc="white", mec="black", ms=4.5, mew=0.7, label="noise-aware S1 searches")
        pts = det_searches("S1", "MG")
        ax.plot([fl(r["k_on"]) for r in pts], [fl(r["k_off"]) for r in pts], "*", mfc="white", mec="black", ms=8, mew=0.7, label="mean-field S1 searches")
        ax.set_xlim(5e-20, 2e-17); ax.set_ylim(0.1, 10); ax.set_title(f"Forecasting under noise, $N_R={NR}$", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=8)
        if k == 1: cb.set_label("NRMSE", fontsize=8.5)
        if k == 0: ax.legend(fontsize=7.6, loc="lower left", labelcolor="white"); panel_label(ax, "(c)", x=-0.2, y=1.08)   # white labels on the dark landscape
        else: panel_label(ax, "(d)", x=-0.2, y=1.08)
    save(fig, "fig6_noise_aware")


def figS8_error_surfaces_noise():
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for i, NR in enumerate((500, 50000)):
        path = RES / f"error_surface_receiver_hyb_NR{NR}.csv"
        if not path.exists(): continue
        allrows = rd(path)
        for j, task in enumerate(TASKS):
            ax = axes[i, j]; rows = [r for r in allrows if r["task"] == task]
            vmin = np.nanmin([fl(r["nrmse_opt_best"]) for r in rows]); vmax = min(1.0, 2.5 * vmin)
            im, Z = landscape_panel(ax, rows, "k_on", "k_off", True, True, vmin, vmax, "$k_{\\mathrm{on}}$ (m$^3$ s$^{-1}$)", "$k_{\\mathrm{off}}$ (s$^{-1}$)" if j == 0 else "")
            kd_lines(ax)
            pts = [r for r in SEARCHES if r["mode"] == "hyb" and int(fl(r["N_R"])) == NR and r["set"] == "S1" and r["task"] == task]
            ax.plot([fl(r["k_on"]) for r in pts], [fl(r["k_off"]) for r in pts], "o", mfc="white", mec="black", ms=4.5, mew=0.7, label="noise-aware S1 searches (five)")
            pts = det_searches("S1", task)
            ax.plot([fl(r["k_on"]) for r in pts], [fl(r["k_off"]) for r in pts], "*", mfc="white", mec="black", ms=8, mew=0.7, label="mean-field S1 searches (ten)")
            ax.set_xlim(5e-20, 2e-17); ax.set_ylim(0.1, 10); ax.set_title(f"{TNAME[task]}, $N_R={NR}$", fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=7)
            if i == 0 and j == 0: ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(w_pad=0.6, h_pad=1.2)
    save(fig, "figS8_error_surfaces_noise")


def figS7_convergence_noise_aware():
    run = rd(RES / "running_min.csv")
    fig, axes = plt.subplots(4, 3, figsize=(10.2, 10.5))
    for i, NR in enumerate((500, 5000, 50000, 500000)):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            for s in SETS:
                rows = sorted([r for r in run if r["mode"] == "hyb" and int(fl(r["N_R"])) == NR and r["set"] == s and r["task"] == task], key=lambda r: int(r["eval_index"]))
                if rows: ax.plot([int(r["eval_index"]) for r in rows], [fl(r["mean_running_min"]) for r in rows], color=SCOL[s], lw=1.2, label=s)
            s0rows = [r for r in S0 if r["mode"] == "hyb" and int(fl(r["N_R"])) == NR and r["task"] == task]
            if s0rows: ax.axhline(min(fl(r["objective_mean"]) for r in s0rows), color="black", lw=0.9, ls="--")
            ax.set_xlim(1, 100); ax.set_title(f"{TNAME[task]}, $N_R={NR}$", fontsize=9)
            if i == 3: ax.set_xlabel("objective evaluations")
            if j == 0: ax.set_ylabel("mean running minimum (five searches)")
            ax.grid(color="0.92", lw=0.6); ax.set_axisbelow(True)
            if i == 0 and j == 2: ax.legend(fontsize=7.5)
    fig.tight_layout(h_pad=1.2)
    save(fig, "figS7_convergence_noise_aware")


def figS9_receptor_number_sweep():
    re_ = reeval(); NRS = sorted(set(int(fl(r["eval_NR"])) for r in re_ if fl(r["eval_NR"]) > 0))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))
    for j, task in enumerate(TASKS):
        ax = axes[0, j]
        for s, col, ls in (("S0", "black", "--"), ("S1", SCOL["S1"], "-"), ("S2", SCOL["S2"], "-"), ("S3", SCOL["S3"], "-"), ("S4", SCOL["S4"], "-")):
            ys = [fl(get_re(re_, "det", 0, s, task, "retrained", nr)["objective"]) for nr in NRS]
            ax.plot(NRS, ys, color=col, ls=ls, lw=1.2, marker="o", ms=3, label=s)
            ysi = [fl(get_re(re_, "det", 0, s, task, "retrained", nr, "integrate")["objective"]) for nr in NRS]
            ax.plot(NRS, ysi, color=col, ls=":", lw=0.9, marker="o", ms=2.5, mfc="white")
        ax.axhline(baseline_best(task), color="0.45", lw=0.9, ls="--", label="input regression")
        ax.set_xscale("log"); ax.set_title(f"{TSHORT[task]}, mean-field optima, readout retrained", fontsize=8.5); ax.set_ylim(0, 1.0); ax.grid(color="0.92", lw=0.6, which="both")
        if j == 0: ax.set_ylabel("NRMSE, retrained (solid sampled, dotted integrating nodes)", fontsize=6.8)
        if j == 2: ax.legend(fontsize=7)
        ax = axes[1, j]
        for s in SETS:
            for dnr, mk in ((500, "v"), (5000, "D"), (50000, "^"), (500000, "s")):
                ys = [get_re(re_, "hyb", dnr, s, task, "retrained", nr) for nr in NRS]
                if all(ys): ax.plot(NRS, [fl(r["objective"]) for r in ys], color=SCOL[s], lw=0.9, marker=mk, ms=3.5, alpha=0.85, label=f"{s}, designed at {dnr}" if j == 2 else None)
        ax.axhline(baseline_best(task), color="0.45", lw=0.9, ls="--")
        ax.set_xscale("log"); ax.set_xlabel("receptor number $N_R$ at evaluation"); ax.set_title(f"{TSHORT[task]}, noise-aware optima, readout retrained", fontsize=8.5); ax.set_ylim(0, 1.0); ax.grid(color="0.92", lw=0.6, which="both")
        if j == 0: ax.set_ylabel("NRMSE, readout retrained at each $N_R$", fontsize=7.5)
        if j == 2: hh, ll = ax.get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=4, fontsize=6.5, handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=1.2, rect=(0, 0.11, 1, 1))
    save(fig, "figS9_receptor_number_sweep")


if __name__ == "__main__":
    for name in WANT:
        globals()[name]()
