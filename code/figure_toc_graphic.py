"""Table of Contents graphic (Wiley wide format, 110 mm by 20 mm), drawn from shapes only.
Left: the tunable MC channel as a physical reservoir with its parameters. Middle: the task sets the receptor operating
point on the binding curve (forecasting in the responsive part, threshold decision in saturation). Right: the same
channel switched between the two tasks in operation, with the receptor number as the price of noise.
Usage: python figure_toc_graphic.py   (writes manuscript/figures/toc_graphic.pdf)"""
import math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 6, "pdf.fonttype": 42})
BLUE, RED, GREEN, PURPLE, ORANGE = "#3B75AF", "#C8553D", "#4F9D5B", "#8E5FB5", "#E8862B"
W, H = 110.0, 20.0   # mm

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")


def ycap(x, y, ang, size, color, lw=0.7):
    a = math.radians(ang); tx, ty = x + size * math.cos(a), y + size * math.sin(a)
    ax.plot([x, tx], [y, ty], color=color, lw=lw, solid_capstyle="round", zorder=6)
    for da in (38, -38):
        aa = a + math.radians(da); ax.plot([tx, tx + 0.6 * size * math.cos(aa)], [ty, ty + 0.6 * size * math.sin(aa)], color=color, lw=lw, solid_capstyle="round", zorder=6)


def text(x, y, s, color="0.25", fs=5.6, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); ax.text(x, y, s, fontsize=fs, color=color, linespacing=1.0, zorder=9, **kw)


# ---------------------------------------------------------------- left: the tunable MC channel (x 1 to 40)
y0 = 10.6
ax.plot(4.6, y0, "o", ms=5.5, color=ORANGE, zorder=4); text(5.0, 15.4, "release", color=ORANGE); text(4.9, 5.4, "$T$, $N_{\\max}$", color=ORANGE, fs=5.2)
rng = np.random.default_rng(7); n = 60
xs = 6.4 + 15.0 * rng.random(n) ** 0.85; frac = (xs - 6.4) / 15.0; ys = y0 + rng.normal(0, 1, n) * (0.6 + 2.5 * frac)
for xx, yy, fr in zip(xs, ys, frac): ax.plot(xx, yy, "o", color="0.3", ms=1.3, alpha=float(0.9 - 0.55 * fr), mec="none", zorder=2)
text(13.8, 16.0, "diffusion", color=GREEN); text(13.8, 4.8, "$d$, $D$", color=GREEN, fs=5.2)
rc = (25.5, y0); R = 3.0
ax.add_patch(Circle(rc, R, fc="0.93", ec="0.45", lw=0.7, zorder=3))
for ang in np.linspace(115, 245, 6):
    a = math.radians(ang); ycap(rc[0] + R * math.cos(a), rc[1] + R * math.sin(a), ang, 1.25, BLUE)
text(25.8, 16.3, "receptors", color=BLUE); text(25.2, 4.2, "$k_{\\mathrm{on}}$, $k_{\\mathrm{off}}$, $N_R$", color=BLUE, fs=5.0)
ax.add_patch(FancyBboxPatch((31.6, 8.5), 7.2, 4.2, boxstyle="round,pad=0.25", fc="white", ec="0.3", lw=0.7, zorder=5))
text(35.2, 10.6, "linear\nreadout", color="0.25", fs=5.2)
ax.annotate("", xy=(31.4, y0), xytext=(28.6, y0), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=5)
text(36.2, 5.0, "$L$, $\\lambda$", color="0.25", fs=5.2)
text(21.0, 1.4, "tunable MC channel as a reservoir", color="0.3", fs=5.1)

# ---------------------------------------------------------------- middle: the task sets the receptor operating point (x 44 to 72)
x_ax0, x_ax1, y_ax0, y_ax1 = 45.5, 70.5, 4.2, 16.4
ax.plot([x_ax0, x_ax1], [y_ax0, y_ax0], color="0.4", lw=0.6); ax.plot([x_ax0, x_ax0], [y_ax0, y_ax1], color="0.4", lw=0.6)
u = np.linspace(-2.2, 2.4, 300); occ = 10 ** u / (1 + 10 ** u)
X = x_ax0 + (u + 2.2) / 4.6 * (x_ax1 - x_ax0); Y = y_ax0 + occ * (y_ax1 - y_ax0 - 0.6)
ax.plot(X, Y, color="black", lw=0.9, zorder=3)
def seg(lo, hi, color):
    m = (u >= lo) & (u <= hi); ax.plot(X[m], Y[m], color=color, lw=2.8, solid_capstyle="butt", zorder=4)
seg(-0.15, 0.55, BLUE); seg(0.85, 1.7, RED)
text(51.5, 12.6, "forecasting", color=BLUE, fs=5.4); text(65.2, 10.4, "threshold\ndecision", color=RED, fs=5.4)
ax.annotate("", xy=(63.6, 17.0), xytext=(58.6, 13.9), arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.6, mutation_scale=6, connectionstyle="arc3,rad=-0.3"), zorder=5)
text(56.6, 17.9, "retune receptors", color="0.35", fs=5.0)
text(70.8, 2.9, "$c/K_D$", color="0.35", fs=5.0, ha="right")
text(57.6, 1.4, "task sets the operating point", color="0.3", fs=5.1)

# ---------------------------------------------------------------- right: one channel, two tasks in operation (x 74 to 110)
xb0, xb1 = 76.0, 108.5
t = np.linspace(0, 1, 400)
sig = 0.5 + 0.28 * np.sin(2 * math.pi * 2.1 * t) + 0.12 * np.sin(2 * math.pi * 5.3 * t + 1.0)
yA0, yA1 = 12.2, 17.4   # forecasting band
ax.plot(xb0 + t * (xb1 - xb0), yA0 + sig * (yA1 - yA0), color="0.55", lw=0.7, zorder=3)
ax.plot(xb0 + t * (xb1 - xb0), yA0 + np.roll(sig, -18) * (yA1 - yA0), color=BLUE, lw=0.9, ls="--", zorder=4)
text(xb0 + 0.3, 18.5, "forecast", color=BLUE, fs=5.2, ha="left")
yB0, yB1 = 4.0, 8.8   # threshold-decision band
thr = 0.62; step = (sig > thr).astype(float)
ax.plot(xb0 + t * (xb1 - xb0), yB0 + sig * (yB1 - yB0), color="0.55", lw=0.7, zorder=3)
ax.plot(xb0 + t * (xb1 - xb0), yB0 + step * (yB1 - yB0), color=RED, lw=0.9, zorder=4)
ax.plot([xb0, xb1], [yB0 + thr * (yB1 - yB0)] * 2, color="0.6", lw=0.5, ls=":", zorder=2)
text(xb0 + 0.3, 10.2, "threshold decision", color=RED, fs=5.2, ha="left")
text(91.6, 1.4, "one channel, online task switching", color="0.3", fs=5.1)
for xx in (42.0, 73.2): ax.plot([xx, xx], [2.6, 18.6], color="0.85", lw=0.5)

fig.savefig(OUT / "toc_graphic.pdf"); print("wrote toc_graphic")
