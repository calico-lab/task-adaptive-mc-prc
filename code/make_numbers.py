"""Numbers and tables of the manuscript, generated from the result files in data/results.

Writes numbers.tex (LaTeX macros used in the prose) and tables.tex (table bodies as macros) into manuscript/, where
main.tex and supporting_information.tex read them. Sections whose result file does not exist are skipped with a note.
Macro names contain letters only. Usage: python make_numbers.py
"""
import csv, json, math, re, statistics as st
from pathlib import Path
from collections import defaultdict
import numpy as np
import simulator as mc

OUT = mc.ROOT / "manuscript"; OUT.mkdir(parents=True, exist_ok=True)
BENCH_DIR = mc.RESULTS / "benchmark_tasks"
TASKS = ["MG", "SINE", "MGCUBED"]; SETS = ["S1", "S2", "S3", "S4"]; ALLSETS = ["S0", "S1", "S2", "S3", "S4"]
TNAME = {"MG": "MG", "SINE": "Sine-to-square", "MGCUBED": "MG-Cubed"}
TWORD = {"MG": "MG", "SINE": "Sine", "MGCUBED": "MGC"}; SWORD = {"S0": "Szero", "S1": "Sone", "S2": "Stwo", "S3": "Sthree", "S4": "Sfour"}
NRWORD = {500: "NRlow", 5000: "NRmid", 50000: "NRhigh", 500000: "NRvhigh", 2000: "NRtwok", 20000: "NRtwentyk", 200000: "NRtwohk"}
NRDESIGN = (500, 5000, 50000, 500000)
macros, numbers, tables = [], {}, {}


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if Path(p).exists() else None


def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def mac(name, value, fmt="{:.3f}"):
    assert name.isalpha(), name
    s = fmt.format(value) if not isinstance(value, str) else value
    macros.append(f"\\newcommand{{\\{name}}}{{{s}}}"); numbers[name] = value


def tab(name, lines):
    assert name.isalpha(), name
    tables[name] = "\n".join(lines)


def pct(a, b):
    return 100.0 * (a - b) / b


def lam_tex(lam):
    e = int(round(math.log10(lam)))
    return f"$10^{{{e}}}$" if e != 0 else "$1$"


def sci(x, digits=2):
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def small3(x):
    """Three-decimal value, or a below-resolution marker for means that would print as 0.000."""
    return "${<}0.001$" if x < 5e-4 else f"{x:.3f}"


searches = rd(BENCH_DIR / "searches.csv"); optima = rd(BENCH_DIR / "optima.csv"); s0 = rd(BENCH_DIR / "s0.csv"); running = rd(BENCH_DIR / "running_min.csv")
det = [r for r in searches if r["mode"] == "det"]


def det_rows(s, t):
    return [r for r in det if r["set"] == s and r["task"] == t]


def s0_det(t, key="objective_mean"):
    rows = [r for r in s0 if r["mode"] == "det" and r["task"] == t]
    best = min(rows, key=lambda r: fl(r["objective_mean"]))
    return fl(best[key]), int(best["L"])


# ---------------------------------------------------------------- counts
mac("NumDetSearches", len(det) - sum(1 for r in det if r["set"] == "S0"), "{:d}")
mac("NumDetEvaluations", sum(int(r["n_evaluations"]) for r in det if r["set"] != "S0"), "{:d}")
hyb = [r for r in searches if r["mode"] == "hyb"]
mac("NumHybSearches", sum(1 for r in hyb if r["set"] != "S0"), "{:d}")
mac("NumParticleRuns", 60, "{:d}")
mac("SineLinearFloor", math.sqrt(1 - 8 / math.pi ** 2))

# ---------------------------------------------------------------- S0 and sets, mean field
for t in TASKS:
    v, L = s0_det(t); vh, _ = s0_det(t, "held_mean")
    mac(f"Szero{TWORD[t]}", v); mac(f"Szero{TWORD[t]}held", vh); mac(f"Szero{TWORD[t]}L", L, "{:d}")
    for L_ in range(1, 6):
        row = [r for r in s0 if r["mode"] == "det" and r["task"] == t and int(r["L"]) == L_][0]
        mac(f"Szero{TWORD[t]}L{'abcde'[L_-1]}", fl(row["objective_mean"]))
    for s in SETS:
        v = [fl(r["objective"]) for r in det_rows(s, t)]; vh = [fl(r["held"]) for r in det_rows(s, t)]
        mac(f"{SWORD[s]}{TWORD[t]}mean", st.mean(v)); mac(f"{SWORD[s]}{TWORD[t]}sd", st.stdev(v), "{:.4f}"); mac(f"{SWORD[s]}{TWORD[t]}min", min(v))
        mac(f"{SWORD[s]}{TWORD[t]}heldmean", st.mean(vh)); mac(f"{SWORD[s]}{TWORD[t]}gainpct", -pct(st.mean(v), s0_det(t)[0]), "{:.0f}")
        mac(f"{SWORD[s]}{TWORD[t]}max", max(v)); mac(f"{SWORD[s]}{TWORD[t]}nearmin", sum(1 for x in v if x <= min(v) + 0.005), "{:d}")
        mac(f"{SWORD[s]}{TWORD[t]}median", st.median(v))
        mac(f"{SWORD[s]}{TWORD[t]}bestgainpct", -pct(min(v), s0_det(t)[0]), "{:.0f}")
for s in SETS:
    g = [-pct(st.mean([fl(r["objective"]) for r in det_rows(s, t)]), s0_det(t)[0]) for t in TASKS]
    mac(f"{SWORD[s]}GainMin", min(g), "{:.0f}"); mac(f"{SWORD[s]}GainMax", max(g), "{:.0f}")
for t in TASKS:
    s1 = st.mean(fl(r["objective"]) for r in det_rows("S1", t)); s4 = st.mean(fl(r["objective"]) for r in det_rows("S4", t)); z = s0_det(t)[0]
    mac(f"SoneOverSfour{TWORD[t]}pct", 100 * (z - s1) / (z - s4), "{:.0f}")
    mac(f"SfourOverSone{TWORD[t]}pct", -pct(s4, s1), "{:.0f}")
v = [fl(r["koff_T"]) for t in TASKS for r in det_rows("S2", t)]
mac("StwoKoffTmin", min(v), "{:.1f}"); mac("StwoKoffTmax", max(v), "{:.1f}")
mac("SoneSpreadMax", max(st.stdev([fl(r["objective"]) for r in det_rows("S1", t)]) for t in TASKS), "{:.4f}")
mac("StwoSpreadMax", max(st.stdev([fl(r["objective"]) for r in det_rows("S2", t)]) for t in TASKS), "{:.4f}")
# sets table body (optimization and held-out block)
lines = []
for t in TASKS:
    for block, key, key0 in (("optimization", "objective", "objective_mean"), ("held-out", "held", "held_mean")):
        cells = [f"{s0_det(t, key0)[0]:.3f}"]
        for s in SETS:
            v = [fl(r[key]) for r in det_rows(s, t)]
            cells.append(f"{st.mean(v):.3f} $\\pm$ {st.stdev(v):.3f}" if st.stdev(v) >= 5e-4 else f"{st.mean(v):.3f}")
        lines.append(f"{TNAME[t]}, {block} & " + " & ".join(cells) + " \\\\")
tab("TabScenarios", lines)
# ranking statements
for t in TASKS:
    means = {s: st.mean(fl(r["objective"]) for r in det_rows(s, t)) for s in SETS}
    order = sorted(SETS, key=lambda s: means[s])
    mac(f"DetOrder{TWORD[t]}", " $<$ ".join(order))

# ---------------------------------------------------------------- optima table (S4 best per task) and descriptors of tuned configurations
lines = []
for t in TASKS:
    o = [r for r in optima if r["mode"] == "det" and r["set"] == "S4" and r["task"] == t][0]
    lines.append(f"{TNAME[t]} & {sci(fl(o['k_on']))} & {fl(o['k_off']):.2f} & {fl(o['T']):.2f} & {1e6*fl(o['distance']):.1f} & {sci(fl(o['D']))} & {fl(o['N_max']):.0f} & {int(float(o['L']))} & {fl(o['objective']):.3f} & {fl(o['koff_T']):.1f} & {fl(o['c_peak_over_KD']):.1f} & {fl(o['tau_D_over_T']):.2f} & {fl(o['occ_max']):.2f} \\\\")
    mac(f"Sfour{TWORD[t]}bestheld", fl(o["held"]))
    for k, name in (("koff_T", "KoffT"), ("c_peak_over_KD", "CpeakKD"), ("tau_D_over_T", "TauDT"), ("occ_max", "OccMax"), ("KD_nM", "KDnM"), ("occ_frac_above_0p8", "FracAbove")):
        mac(f"BestSfour{TWORD[t]}{name}", fl(o[k]), "{:.2f}" if k != "KD_nM" else "{:.2g}")
    mac(f"BestSfour{TWORD[t]}dum", 1e6 * fl(o["distance"]), "{:.1f}"); mac(f"BestSfour{TWORD[t]}T", fl(o["T"]), "{:.2f}")
tab("TabOptima", lines)
# tuned-configuration descriptors over S1, S3, S4 bests (30 configurations per task)
for t in TASKS:
    rows = [r for s in ("S1", "S3", "S4") for r in det_rows(s, t)]
    for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.1f}"), ("occ_frac_above_0p8", "FracAbove", "{:.2f}"), ("KD_nM", "KDnM", "{:.2g}"), ("occ_max", "OccMax", "{:.2f}"), ("c_mean_over_KD", "CmeanKD", "{:.1f}")):
        v = [fl(r[k]) for r in rows]
        mac(f"Tuned{name}{TWORD[t]}", st.median(v), fmt); mac(f"Tuned{name}{TWORD[t]}min", min(v), fmt); mac(f"Tuned{name}{TWORD[t]}max", max(v), fmt)
    v = [fl(r["tau_D_over_T"]) for r in det_rows("S4", t)]
    mac(f"SfourTauDT{TWORD[t]}min", min(v), "{:.2f}"); mac(f"SfourTauDT{TWORD[t]}max", max(v), "{:.2f}"); mac(f"SfourTauDT{TWORD[t]}median", st.median(v), "{:.2f}")
    v = [fl(r["c_peak_over_KD"]) for r in det_rows("S4", t)]; mac(f"SfourCpeakKD{TWORD[t]}median", st.median(v), "{:.0f}")
    Ls = [int(float(r["L"])) for s in SETS for r in det_rows(s, t)]
    mac(f"TunedLfive{TWORD[t]}", sum(1 for L in Ls if L == 5), "{:d}"); mac(f"TunedLcount{TWORD[t]}", len(Ls), "{:d}")
allL = [int(float(r["L"])) for r in det if r["set"] != "S0"]
mac("DetLfiveCount", sum(1 for L in allL if L == 5), "{:d}"); mac("DetLCount", len(allL), "{:d}")
# boundary statistics of S4 and S1 optima
def at_bound(v, lo, hi, log):
    tol = 0.02 if not log else 0.05
    if log: return v <= lo * (1 + tol) or v >= hi * (1 - tol)
    return v <= lo + tol * (hi - lo) or v >= hi - tol * (hi - lo)
nb = defaultdict(int)
for r in det:
    if r["set"] == "S4":
        for k in mc.PARAM_NAMES:
            kind, lo, hi = mc.DOMAIN[k]
            nb[k] += at_bound(fl(r[k]), lo, hi, kind == "log")
mac("SfourKoffAtBound", nb["k_off"], "{:d}"); mac("SfourTAtBound", nb["T"], "{:d}"); mac("SfourDistAtBound", nb["distance"], "{:d}"); mac("SfourNmaxAtBound", nb["N_max"], "{:d}"); mac("SfourKonAtBound", nb["k_on"], "{:d}")
mac("SoneMGKoffAtBound", sum(at_bound(fl(r["k_off"]), 0.1, 10, True) for r in det_rows("S1", "MG")), "{:d}")
nom = mc.evaluate("MG", dict(mc.NOMINAL, L=5))
mac("NominalKoffT", nom["koff_T"], "{:.2f}"); mac("NominalCpeakKD", nom["c_peak_over_KD"], "{:.1f}"); mac("NominalTauDT", nom["tau_D_over_T"], "{:.2f}"); mac("NominalKDnM", nom["KD_nM"], "{:.2f}")
mac("NominalOccMax", nom["occ_max"], "{:.2f}")

# ---------------------------------------------------------------- convergence
def reach(task, s, frac=0.01):
    rows = sorted([r for r in running if r["mode"] == "det" and r["set"] == s and r["task"] == task], key=lambda r: int(r["eval_index"]))
    m = np.array([fl(r["mean_running_min"]) for r in rows]); final = m[-1]
    return int(np.argmax(m <= final * (1 + frac))) + 1, m
for t in TASKS:
    for s in SETS:
        n, m = reach(t, s)
        mac(f"Budget{SWORD[s]}{TWORD[t]}reach", n, "{:d}"); mac(f"Budget{SWORD[s]}{TWORD[t]}atTwenty", m[19]); mac(f"Budget{SWORD[s]}{TWORD[t]}atHundred", m[-1])
        mac(f"Budget{SWORD[s]}{TWORD[t]}atTen", m[9])

# ---------------------------------------------------------------- landscapes
for name, xa, ya, word in (("receiver", "k_on", "k_off", "Receiver"), ("transmitter", "T", "N_max", "Transmitter"), ("channel", "distance", "D", "Channel")):
    rows = rd(BENCH_DIR / f"error_surface_{name}.csv")
    if rows is None: continue
    for t in TASKS:
        sub = [r for r in rows if r["task"] == t]; v = np.array([fl(r["nrmse_opt_best"]) for r in sub]); ok = np.isfinite(v)
        b = sub[int(np.nanargmin(v))]; vmin = np.nanmin(v)
        mac(f"Land{word}Min{TWORD[t]}", vmin); mac(f"Land{word}Near{TWORD[t]}pct", 100 * np.mean(v[ok] <= 1.1 * vmin), "{:.0f}")
        mac(f"Land{word}Max{TWORD[t]}", np.nanmax(v), "{:.2f}")
        if name == "receiver":
            mac(f"LandReceiverBestKD{TWORD[t]}", fl(b["KD_nM"]), "{:.2g}"); mac(f"LandReceiverBestKoffT{TWORD[t]}", fl(b["koff_T"]), "{:.1f}"); mac(f"LandReceiverBestKoff{TWORD[t]}", fl(b["k_off"]), "{:.2g}")
        if name == "transmitter":
            nomv = [fl(r["nrmse_opt_best"]) for r in sub if abs(fl(r["T"]) - 1.25) < 0.04 and abs(fl(r["N_max"]) - 10100) < 500]
            mac(f"LandTransmitterGain{TWORD[t]}pct", -pct(vmin, s0_det(t)[0]), "{:.0f}")
            mac(f"LandTransmitterBestT{TWORD[t]}", fl(b["T"]), "{:.2f}"); mac(f"LandTransmitterBestNmax{TWORD[t]}", fl(b["N_max"]), "{:.0f}")
        if name == "channel":
            mac(f"LandChannelBestD{TWORD[t]}", fl(b["D"]), "{:.1e}"); mac(f"LandChannelBestDist{TWORD[t]}", 1e6 * fl(b["distance"]), "{:.1f}")
            mac(f"LandChannelBestTauDT{TWORD[t]}", fl(b["tau_D_over_T"]), "{:.2f}")
    lines = []
    for t in TASKS:
        sub = [r for r in rows if r["task"] == t]; v = np.array([fl(r["nrmse_opt_best"]) for r in sub]); b = sub[int(np.nanargmin(v))]
        if name == "receiver": loc = f"{sci(fl(b['k_on']))} & {fl(b['k_off']):.2f}"
        elif name == "transmitter": loc = f"{fl(b['T']):.2f} & {fl(b['N_max']):.0f}"
        else: loc = f"{1e6*fl(b['distance']):.1f} & {sci(fl(b['D']))}"
        lines.append(f"{TNAME[t]} & {loc} & {int(float(b['L_best']))} & {np.nanmin(v):.3f} & {np.nanmax(v):.3f} & {100*np.mean(v[np.isfinite(v)] <= 1.1*np.nanmin(v)):.0f} \\\\")
    tab(f"TabLandscape{word}", lines)

# ---------------------------------------------------------------- reuse (mean field)
reuse = rd(BENCH_DIR / "reuse.csv")
if reuse:
    d = defaultdict(list); dh = defaultdict(list)
    for r in reuse:
        if r["mode"] == "det":
            d[(r["set"], r["source_task"], r["target_task"])].append(fl(r["objective"])); dh[(r["set"], r["source_task"], r["target_task"])].append(fl(r["held"]))
    gap = {}; gaph = {}; above = 0; lines = []; linesh = []
    for s in SETS:
        for tgt in TASKS:
            ref = st.median(d[(s, tgt, tgt)]); refh = st.median(dh[(s, tgt, tgt)])
            for src in TASKS:
                if src == tgt: continue
                m = st.median(d[(s, src, tgt)]); g = pct(m, ref); gap[(s, src, tgt)] = g
                mh = st.median(dh[(s, src, tgt)]); gaph[(s, src, tgt)] = pct(mh, refh)
                flag = m > s0_det(tgt)[0]; above += flag
                q1, q3 = np.percentile(d[(s, src, tgt)], [25, 75]); q1h, q3h = np.percentile(dh[(s, src, tgt)], [25, 75])
                lines.append(f"{s} & {TNAME[src]} & {TNAME[tgt]} & ${m:.3f}\\,[{q1:.3f},\\,{q3:.3f}]$ & {ref:.3f} & {g:.1f} & {'yes' if flag else ''} \\\\")
                linesh.append(f"{s} & {TNAME[src]} & {TNAME[tgt]} & ${mh:.3f}\\,[{q1h:.3f},\\,{q3h:.3f}]$ & {refh:.3f} & {gaph[(s, src, tgt)]:.1f} & {'yes' if mh > s0_det(tgt, 'held_mean')[0] else ''} \\\\")
    tab("TabReuseScenarios", lines); tab("TabReuseScenariosHeld", linesh)
    mac("ReuseGapMin", min(gap.values()), "{:.1f}"); mac("ReuseGapMax", max(gap.values()), "{:.0f}"); mac("ReusePositive", sum(g > 0 for g in gap.values()), "{:d}")
    mac("ReuseAboveSzero", above, "{:d}"); mac("ReuseCombos", len(gap), "{:d}")
    mac("ReuseHeldPositive", sum(g > 0 for g in gaph.values()), "{:d}"); mac("ReuseHeldGapMax", max(gaph.values()), "{:.0f}")
    mg_pair = [gap[(s, a, b)] for s in SETS for a, b in (("MG", "MGCUBED"), ("MGCUBED", "MG"))]
    mac("ReuseMGpairMin", min(mg_pair), "{:.0f}"); mac("ReuseMGpairMax", max(mg_pair), "{:.0f}")
    v = [gap[(s, "SINE", "MG")] for s in SETS]; mac("ReuseSineOnMGMin", min(v), "{:.0f}"); mac("ReuseSineOnMGMax", max(v), "{:.0f}")
    v = [gap[(s, "MG", "SINE")] for s in SETS] + [gap[(s, "MGCUBED", "SINE")] for s in SETS]; mac("ReuseOnSineMin", min(v), "{:.0f}"); mac("ReuseOnSineMax", max(v), "{:.0f}")
    for s in SETS:
        mac(f"ReuseMeanGap{SWORD[s]}", st.mean(gap[(s, a, b)] for a in TASKS for b in TASKS if a != b), "{:.0f}")
else:
    print("reuse.csv missing, reuse macros skipped")

# ---------------------------------------------------------------- baselines
bt = rd(BENCH_DIR / "baseline_taps.csv")
if bt:
    for r in bt:
        mac(f"Taps{TWORD[r['task']]}{'abcdefg'[[1,2,3,5,10,20,50].index(int(r['taps']))]}", fl(r["objective"]))
    lines = []
    for t in TASKS:
        cells = [f"{fl(r['objective']):.3f}" for r in bt if r["task"] == t]
        lines.append(f"{TNAME[t]} & " + " & ".join(cells) + " \\\\")
    tab("TabBaselineTaps", lines)
    for t in TASKS:
        mac(f"TapsBest{TWORD[t]}", min(fl(r["objective"]) for r in bt if r["task"] == t))
bl1 = rd(BENCH_DIR / "baseline_L1.csv")
if bl1:
    for r in bl1:
        if r["design_mode"] == "det" and r["set"] in ("S1", "S4"):
            mac(f"Lone{SWORD[r['set']]}{TWORD[r['task']]}det", fl(r["det_L1"]))

# ---------------------------------------------------------------- particle validation
pv = rd(BENCH_DIR / "particle_validation.csv")
if pv:
    lines = []
    diffs_f, diffs_r = [], []
    for r in pv:
        lines.append(f"{TNAME[r['task']]} & {r['set']} & {fl(r['meanfield']):.3f} & {fl(r['smoldyn_raw_mean']):.3f} $\\pm$ {fl(r['smoldyn_raw_sd']):.3f} & {fl(r['hybrid_raw_mean']):.3f} $\\pm$ {fl(r['hybrid_raw_sd']):.3f} & {fl(r['smoldyn_filt_mean']):.3f} $\\pm$ {fl(r['smoldyn_filt_sd']):.3f} & {fl(r['hybrid_filt_mean']):.3f} $\\pm$ {fl(r['hybrid_filt_sd']):.3f} & {fl(r['hybrid_retrained']):.3f} \\\\")
        diffs_f.append(fl(r["hybrid_filt_mean"]) - fl(r["smoldyn_filt_mean"])); diffs_r.append(fl(r["hybrid_raw_mean"]) - fl(r["smoldyn_raw_mean"]))
    tab("TabParticle", lines)
    mac("PartValidFiltMaxAbsDiff", max(abs(x) for x in diffs_f), "{:.3f}"); mac("PartValidRawMaxAbsDiff", max(abs(x) for x in diffs_r), "{:.3f}")
    mac("PartValidFiltMeanDiff", st.mean(diffs_f), "{:+.3f}"); mac("PartValidRawMeanDiff", st.mean(diffs_r), "{:+.3f}")
    mac("PartValidR", int(float(pv[0]["R"])), "{:d}"); mac("ParticleValidR", int(float(pv[0]["R"])), "{:d}")
    from scipy.stats import spearmanr, pearsonr
    sm = [fl(r["smoldyn_filt_mean"]) for r in pv] + [fl(r["smoldyn_raw_mean"]) for r in pv]; hy = [fl(r["hybrid_filt_mean"]) for r in pv] + [fl(r["hybrid_raw_mean"]) for r in pv]
    mac("PartValidPearson", pearsonr(sm, hy)[0], "{:.3f}"); mac("PartValidSpearman", spearmanr(sm, hy)[0], "{:.2f}")
    mac("PartValidSmolSdMax", max(max(fl(r["smoldyn_filt_sd"]), fl(r["smoldyn_raw_sd"])) for r in pv), "{:.3f}")
    mac("PartRawMinMG", min(fl(r["smoldyn_raw_mean"]) for r in pv if r["task"] == "MG")); mac("PartRawMaxMGC", max(fl(r["smoldyn_raw_mean"]) for r in pv if r["task"] == "MGCUBED"))
    mac("PartFiltMin", min(fl(r["smoldyn_filt_mean"]) for r in pv)); mac("PartFiltMax", max(fl(r["smoldyn_filt_mean"]) for r in pv))
    mac("PartMeanfieldMin", min(fl(r["meanfield"]) for r in pv)); mac("PartMeanfieldMax", max(fl(r["meanfield"]) for r in pv))

# ---------------------------------------------------------------- re-evaluation under noise
re_ = rd(BENCH_DIR / "reevaluation.csv")
if re_:
    R_re = int(float([r for r in re_ if fl(r["eval_NR"]) > 0][0]["R"])); mac("ReevalR", R_re, "{:d}")
    NRS = sorted(set(int(float(r["eval_NR"])) for r in re_ if fl(r["eval_NR"]) > 0))
    def get(dm, dnr, s, t, proto, nr, node="sample"):
        for r in re_:
            if r["design_mode"] == dm and int(float(r["design_NR"])) == dnr and r["set"] == s and r["task"] == t and r["protocol"] == proto and int(float(r["eval_NR"])) == nr and r["node"] == node:
                return r
        return None
    # mean-field optima under noise (designed and retrained) at every N_R
    for s in ALLSETS:
        for t in TASKS:
            for nr in NRS:
                for proto, pw in (("designed", "Des"), ("retrained", "Ret")):
                    r = get("det", 0, s, t, proto, nr)
                    if r and nr in NRWORD:
                        mac(f"MF{SWORD[s]}{TWORD[t]}{pw}{NRWORD[nr]}", fl(r["objective"])); mac(f"MF{SWORD[s]}{TWORD[t]}{pw}{NRWORD[nr]}sd", fl(r["opt_sd"]), "{:.3f}")
                        if proto == "retrained":
                            mac(f"MF{SWORD[s]}{TWORD[t]}{pw}{NRWORD[nr]}L", int(float(r["L"])), "{:d}"); mac(f"MF{SWORD[s]}{TWORD[t]}{pw}{NRWORD[nr]}lam", lam_tex(fl(r["lam"])))
                r = get("det", 0, s, t, "retrained", nr, "integrate")
                if r and nr in NRWORD: mac(f"MF{SWORD[s]}{TWORD[t]}Int{NRWORD[nr]}", fl(r["objective"]))
    # noise-aware optima re-evaluated at their design N_R (designed protocol) and at other N_R (retrained)
    for s in ALLSETS:
        for t in TASKS:
            for dnr in NRDESIGN:
                r = get("hyb", dnr, s, t, "designed", dnr)
                if r:
                    mac(f"NA{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}", fl(r["objective"])); mac(f"NA{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}sd", fl(r["opt_sd"]), "{:.3f}")
                    mac(f"NA{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}held", fl(r["held"])); mac(f"NA{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}L", int(float(r["L"])), "{:d}"); mac(f"NA{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}lam", lam_tex(fl(r["lam"])))
                    mf = get("det", 0, s, t, "retrained", dnr)
                    if mf: mac(f"NAgain{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}pct", -pct(fl(r["objective"]), fl(mf["objective"])), "{:.0f}")
                    z = get("det", 0, "S0", t, "retrained", dnr)
                    if z: mac(f"NAoverSzero{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}pct", -pct(fl(r["objective"]), fl(z["objective"])), "{:.0f}")
                for nr in NRS:
                    r = get("hyb", dnr, s, t, "retrained", nr)
                    if r and nr in NRWORD: mac(f"NA{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}at{NRWORD[nr]}", fl(r["objective"]))
    # ratios of the noise-aware optima to the mean-field optimum of the same set, and decay of the retrained mean-field optima
    for s in ("S1", "S4"):
        for t in TASKS:
            mfmin = min(fl(r["objective"]) for r in det_rows(s, t))
            for dnr in NRDESIGN:
                r = get("hyb", dnr, s, t, "designed", dnr)
                if r: mac(f"NAoverMF{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}", fl(r["objective"]) / mfmin, "{:.1f}")
            lo = get("det", 0, s, t, "retrained", 500); hi = get("det", 0, s, t, "retrained", 500000)
            if lo and hi: mac(f"MFdecay{SWORD[s]}{TWORD[t]}", fl(lo["objective"]) / fl(hi["objective"]), "{:.1f}")
    mac("SqrtThousand", math.sqrt(1000.0), "{:.0f}")
    # table: noise-aware results per task and N_R
    lines = []
    for t in TASKS:
        for dnr in NRDESIGN:
            z = get("det", 0, "S0", t, "retrained", dnr); mf4 = get("det", 0, "S4", t, "retrained", dnr); mf1 = get("det", 0, "S1", t, "retrained", dnr)
            cells = [f"{fl(z['objective']):.3f}" if z else "", f"{fl(mf1['objective']):.3f}" if mf1 else "", f"{fl(mf4['objective']):.3f}" if mf4 else ""]
            for s in SETS:
                r = get("hyb", dnr, s, t, "designed", dnr)
                cells.append(f"{fl(r['objective']):.3f} $\\pm$ {fl(r['opt_sd']):.3f}" if r else "")
            lines.append(f"{TNAME[t]} & {dnr} & " + " & ".join(cells) + " \\\\")
    tab("TabNoiseAware", lines)
    # crossing of the input-tap baseline
    if bt:
        for t in TASKS:
            base5 = [fl(r["objective"]) for r in bt if r["task"] == t and int(r["taps"]) == 5][0]
            best_base = min(fl(r["objective"]) for r in bt if r["task"] == t)
            for s in ("S1", "S4"):
                cross = None
                for nr in NRS:
                    r = get("det", 0, s, t, "retrained", nr)
                    if r and fl(r["objective"]) < best_base:
                        cross = nr; break
                mac(f"CrossTaps{SWORD[s]}{TWORD[t]}", f"{cross}" if cross else "none")
    # descriptors of the noise-aware optima (from optima.csv)
    for s in SETS:
        for t in TASKS:
            for dnr in NRDESIGN:
                o = [r for r in optima if r["mode"] == "hyb" and int(float(r["N_R"])) == dnr and r["set"] == s and r["task"] == t]
                if o:
                    o = o[0]
                    for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}"), ("occ_max", "OccMax", "{:.2f}"), ("tau_D_over_T", "TauDT", "{:.2f}"), ("KD_nM", "KDnM", "{:.2g}"), ("k_off", "Koff", "{:.2g}")):
                        mac(f"NAdesc{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}{name}", fl(o[k]), fmt)
                    mac(f"NAdesc{SWORD[s]}{TWORD[t]}{NRWORD[dnr]}Lvalues", " ".join(o["L_values"].split()))
    # table of noise-aware optima parameters (S1 and S4) per N_R
    lines = []
    for t in TASKS:
        for s in ("S1", "S4"):
            for dnr in NRDESIGN:
                o = [r for r in optima if r["mode"] == "hyb" and int(float(r["N_R"])) == dnr and r["set"] == s and r["task"] == t]
                if not o: continue
                o = o[0]; r = get("hyb", dnr, s, t, "designed", dnr)
                lines.append(f"{TNAME[t]} & {s} & {dnr} & {sci(fl(o['k_on']))} & {fl(o['k_off']):.2f} & {fl(o['T']):.2f} & {1e6*fl(o['distance']):.1f} & {sci(fl(o['D']))} & {fl(o['N_max']):.0f} & {int(float(o['L']))} & {lam_tex(fl(o['lam_best']))} & {fl(o['koff_T']):.1f} & {fl(o['c_peak_over_KD']):.2g} & {fl(r['objective']) if r else fl(o['objective']):.3f} \\\\")
    tab("TabNoiseAwareOptima", lines)

# ---------------------------------------------------------------- reuse under noise
if reuse:
    hr = [r for r in reuse if r["mode"] == "hyb"]
    if hr:
        gaps = {}
        for nr in NRDESIGN:
            for s in SETS:
                for tgt in TASKS:
                    ref = [fl(r["objective"]) for r in hr if int(float(r["N_R"])) == nr and r["set"] == s and r["source_task"] == tgt and r["target_task"] == tgt]
                    for src in TASKS:
                        if src == tgt: continue
                        v = [fl(r["objective"]) for r in hr if int(float(r["N_R"])) == nr and r["set"] == s and r["source_task"] == src and r["target_task"] == tgt]
                        if ref and v: gaps[(nr, s, src, tgt)] = pct(v[0], ref[0])
        for nr in NRDESIGN:
            g = [v for k, v in gaps.items() if k[0] == nr]
            if g:
                mac(f"NAreuseGapMin{NRWORD[nr]}", min(g), "{:.0f}"); mac(f"NAreuseGapMax{NRWORD[nr]}", max(g), "{:.0f}"); mac(f"NAreusePositive{NRWORD[nr]}", sum(x > 0 for x in g), "{:d}"); mac(f"NAreuseCount{NRWORD[nr]}", len(g), "{:d}")
                mac(f"NAreuseMedianGap{NRWORD[nr]}", st.median(g), "{:.0f}")
        lines = []
        for nr in NRDESIGN:
            for s in SETS:
                for tgt in TASKS:
                    for src in TASKS:
                        if src == tgt or (nr, s, src, tgt) not in gaps: continue
                        v = [r for r in hr if int(float(r["N_R"])) == nr and r["set"] == s and r["source_task"] == src and r["target_task"] == tgt][0]
                        ref = [r for r in hr if int(float(r["N_R"])) == nr and r["set"] == s and r["source_task"] == tgt and r["target_task"] == tgt][0]
                        lines.append(f"{nr} & {s} & {TNAME[src]} & {TNAME[tgt]} & {fl(v['objective']):.3f} $\\pm$ {fl(v['opt_sd']):.3f} & {fl(ref['objective']):.3f} $\\pm$ {fl(ref['opt_sd']):.3f} & {gaps[(nr, s, src, tgt)]:.0f} \\\\")
        tab("TabReuseNoise", lines)
        mac("ReuseR", int(float(hr[0]["R"])), "{:d}")

# ---------------------------------------------------------------- robustness
rb = rd(BENCH_DIR / "robustness.csv")
if rb:
    dd = [abs(fl(r["det_fine"]) - fl(r["det_default"])) for r in rb] + [abs(fl(r["det_full_history"]) - fl(r["det_default"])) for r in rb]
    mac("RobustDetMaxDiff", max(dd), "{:.4f}"); mac("RobustDetFineMax", max(abs(fl(r["det_fine"]) - fl(r["det_default"])) for r in rb), "{:.4f}"); mac("RobustDetHistMax", max(abs(fl(r["det_full_history"]) - fl(r["det_default"])) for r in rb), "{:.4f}")
    hd = [abs(fl(r["hyb_fine"]) - fl(r["hyb_default"])) for r in rb] + [abs(fl(r["hyb_full_history"]) - fl(r["hyb_default"])) for r in rb] + [abs(fl(r["hyb_exact_binomial"]) - fl(r["hyb_default"])) for r in rb]
    mac("RobustHybMaxDiff", max(hd), "{:.3f}"); mac("RobustHybSdMax", max(fl(r["hyb_default_sd"]) for r in rb), "{:.3f}"); mac("RobustHybExactMax", max(abs(fl(r["hyb_exact_binomial"]) - fl(r["hyb_default"])) for r in rb), "{:.3f}")
    DESIGN = {"det": "mean-field", "hyb": "noise-aware"}   # design labels of Table S2
    lines = [f"{DESIGN[r['design_mode']]} & {int(float(r['design_NR'])) or ''} & {r['set']} & {TNAME[r['task']]} & {fl(r['det_default']):.4f} & {fl(r['det_fine']):.4f} & {fl(r['det_full_history']):.4f} & {fl(r['hyb_default']):.3f} $\\pm$ {fl(r['hyb_default_sd']):.3f} & {fl(r['hyb_fine']):.3f} & {fl(r['hyb_full_history']):.3f} & {fl(r['hyb_exact_binomial']):.3f} \\\\" for r in rb]
    tab("TabRobustness", lines)

# ---------------------------------------------------------------- SI tables: every search, S0 enumeration, N_R sweep
lines = []
for r in sorted(det, key=lambda r: (ALLSETS.index(r["set"]), TASKS.index(r["task"]), int(r["search_seed"]))):
    if r["set"] == "S0": continue
    lines.append(f"{r['set']} & {TNAME[r['task']]} & {r['search_seed']} & {sci(fl(r['k_on']))} & {fl(r['k_off']):.2f} & {fl(r['T']):.2f} & {1e6*fl(r['distance']):.1f} & {sci(fl(r['D']))} & {fl(r['N_max']):.0f} & {int(float(r['L']))} & {fl(r['objective']):.4f} & {fl(r['held']):.4f} & {fl(r['koff_T']):.1f} & {fl(r['c_peak_over_KD']):.1f} & {fl(r['tau_D_over_T']):.2f} \\\\")
tab("TabAllSearchesDet", lines)
lines = []
for r in sorted(hyb, key=lambda r: (int(float(r["N_R"])), ALLSETS.index(r["set"]), TASKS.index(r["task"]), int(r["search_seed"]))):
    if r["set"] == "S0": continue
    lines.append(f"{int(float(r['N_R']))} & {r['set']} & {TNAME[r['task']]} & {r['search_seed']} & {sci(fl(r['k_on']))} & {fl(r['k_off']):.2f} & {fl(r['T']):.2f} & {1e6*fl(r['distance']):.1f} & {sci(fl(r['D']))} & {fl(r['N_max']):.0f} & {int(float(r['L']))} & {lam_tex(fl(r['lam_best']))} & {fl(r['objective']):.3f} & {fl(r['held']):.3f} & {fl(r['koff_T']):.1f} & {fl(r['c_peak_over_KD']):.2g} \\\\")
tab("TabAllSearchesHyb", lines)
lines = []
for t in TASKS:
    cells = []
    for L_ in range(1, 6):
        row = [r for r in s0 if r["mode"] == "det" and r["task"] == t and int(r["L"]) == L_][0]
        cells.append(f"{fl(row['objective_mean']):.4f} / {fl(row['held_mean']):.4f}")
    lines.append(f"{TNAME[t]} & " + " & ".join(cells) + " \\\\")
tab("TabSzeroEnum", lines)
if re_:
    lines = []
    for s in ALLSETS:
        for t in TASKS:
            for proto in ("designed", "retrained"):
                cells = []
                for nr in NRS:
                    r = get("det", 0, s, t, proto, nr)
                    cells.append(f"{fl(r['objective']):.3f}" if r else "n.e.")
                mfv = get("det", 0, s, t, "designed", 0)
                lines.append(f"{s} & {TNAME[t]} & {proto} & {fl(mfv['objective']) if mfv else float('nan'):.3f} & " + " & ".join(cells) + " \\\\")
    tab("TabNRsweepMF", lines)
    lines = []
    for dnr in NRDESIGN:
        for s in SETS:
            for t in TASKS:
                cells = []
                for nr in NRS:
                    r = get("hyb", dnr, s, t, "retrained", nr)
                    cells.append(f"{fl(r['objective']):.3f}" if r else "n.e.")
                mfv = get("hyb", dnr, s, t, "designed", 0)
                lines.append(f"{dnr} & {s} & {TNAME[t]} & {fl(mfv['objective']) if mfv else float('nan'):.3f} & " + " & ".join(cells) + " \\\\")
    tab("TabNRsweepNA", lines)
    mac("NRsweepValues", ", ".join(str(n) for n in NRS))

# ---------------------------------------------------------------- particle simulations and L = 1 table
smol = rd(mc.ROOT / "data" / "particle_simulations" / "smoldyn_60_seed_level_values.csv")
if smol:
    lines = []
    for sid in sorted(set(r["selection_id"] for r in smol), key=lambda x: (x.split("_")[0], TASKS.index(x.split("_", 1)[1]))):
        rows = sorted([r for r in smol if r["selection_id"] == sid], key=lambda r: int(r["smoldyn_seed"]))
        raw = ", ".join(f"{fl(r['raw_smoldyn_nrmse']):.3f}" for r in rows); filt = ", ".join(f"{fl(r['filtered_smoldyn_nrmse']):.3f}" for r in rows)
        lines.append(f"{rows[0]['scenario']} & {TNAME[rows[0]['task']]} & {fl(rows[0]['bo_nrmse']):.3f} & {raw} & {filt} \\\\")
    tab("TabParticleRuns", lines)
if bl1:
    lines = []
    for r in bl1:
        if r["design_mode"] == "det":
            lines.append(f"{r['set']} & {TNAME[r['task']]} & {int(float(r['design_L']))} & {fl(r['design_objective']):.3f} & {fl(r['det_L1']):.3f} & {fl(r['hyb_L1']):.3f} $\\pm$ {fl(r['hyb_L1_sd']):.3f} \\\\")
    tab("TabLone", lines)
    v = [(fl(r["det_L1"]) - fl(r["design_objective"])) / fl(r["design_objective"]) * 100 for r in bl1 if r["design_mode"] == "det" and r["set"] != "S0"]
    mac("LoneLossMinpct", min(v), "{:.0f}"); mac("LoneLossMaxpct", max(v), "{:.0f}")

# ================================================================ case study, wide domain, and minimax (compromise) searches
GROUPS_DIR = mc.RESULTS / "case_study_minimax_wide_domain"
CASE = ("GLUF", "GLUE"); CWORD = {"GLUF": "Gluf", "GLUE": "Glue"}; CNAME = {"GLUF": "Biomarker forecast", "GLUE": "Threshold decision"}
opt_grp = rd(GROUPS_DIR / "optima.csv"); se_grp = rd(GROUPS_DIR / "searches.csv"); re_grp = rd(GROUPS_DIR / "reevaluation.csv"); bt_grp = rd(GROUPS_DIR / "baseline_taps.csv")


def o_grp(group, mode, NR, s, t):
    for r in (opt_grp or []):
        if r["group"] == group and r["mode"] == mode and int(float(r["N_R"])) == NR and r["set"] == s and r["task"] == t: return r
    return None


def re_grp_get(kind, dmode, dnr, s, dtask, etask, enr, proto):
    for r in (re_grp or []):
        if r["kind"] == kind and r["design_mode"] == dmode and int(float(r["design_NR"])) == dnr and r["set"] == s and r["design_task"] == dtask and r["eval_task"] == etask and int(float(r["eval_NR"])) == enr and r["protocol"] == proto: return r
    return None


if opt_grp:
    # ---- case study
    for t in CASE:
        for s in ("S0", "S1", "S4"):
            o = o_grp("case", "det", 0, s, t)
            if o:
                mac(f"Case{SWORD[s]}{CWORD[t]}min", fl(o["objective"])); mac(f"Case{SWORD[s]}{CWORD[t]}mean", fl(o["obj_mean"])); mac(f"Case{SWORD[s]}{CWORD[t]}held", fl(o["held"]))
                for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}"), ("tau_D_over_T", "TauDT", "{:.2f}"), ("occ_frac_above_0p8", "FracAbove", "{:.2f}"), ("KD_nM", "KDnM", "{:.2g}")):
                    mac(f"Case{SWORD[s]}{CWORD[t]}{name}", fl(o[k]), fmt)
                mac(f"Case{SWORD[s]}{CWORD[t]}L", int(float(o["L"])), "{:d}")
            for NR in (500, 50000):
                o = o_grp("case", "hyb", NR, s, t)
                if o:
                    mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}search", fl(o["objective"]))
                    for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}"), ("tau_D_over_T", "TauDT", "{:.2f}")):
                        mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}{name}", fl(o[k]), fmt)
                    mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}lam", lam_tex(fl(o["lam_best"]))); mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}L", int(float(o["L"])), "{:d}")
                    r = re_grp_get("case", "hyb", NR, s, t, t, NR, "designed")
                    if r: mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}", fl(r["objective"])); mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}sd", fl(r["opt_sd"]), "{:.3f}")
                r = re_grp_get("case", "det", 0, s, t, t, NR, "retrained")
                if r: mac(f"CaseMF{SWORD[s]}{CWORD[t]}Ret{NRWORD[NR]}", fl(r["objective"]))
        if bt_grp:
            mac(f"TapsBest{CWORD[t]}", min(fl(r["objective"]) for r in bt_grp if r["task"] == t)); mac(f"Taps{CWORD[t]}d", [fl(r["objective"]) for r in bt_grp if r["task"] == t and int(r["taps"]) == 5][0])
    if bt_grp:
        tab("TabBaselineTapsCase", [f"{CNAME[t]} & " + " & ".join(f"{fl(r['objective']):.3f}" for r in bt_grp if r["task"] == t) + " \\\\" for t in CASE])
    for t in CASE:
        z = o_grp("case", "det", 0, "S0", t)
        for s in ("S1", "S4"):
            o = o_grp("case", "det", 0, s, t)
            if o and z: mac(f"Case{SWORD[s]}{CWORD[t]}gainpct", -pct(fl(o["objective"]), fl(z["objective"])), "{:.0f}")
        for NR in (500, 50000):
            z = re_grp_get("case", "det", 0, "S0", t, t, NR, "retrained")
            for s in ("S1", "S4"):
                r = re_grp_get("case", "hyb", NR, s, t, t, NR, "designed")
                if r and z: mac(f"CaseNA{SWORD[s]}{CWORD[t]}{NRWORD[NR]}gainpct", -pct(fl(r["objective"]), fl(z["objective"])), "{:.0f}")
    lines = []
    for t in CASE:
        for s in ("S0", "S1", "S4"):
            o = o_grp("case", "det", 0, s, t)
            if not o: continue
            cells = [f"{fl(o['objective']):.3f}"]
            for NR in (500, 50000):
                r = re_grp_get("case", "det", 0, s, t, t, NR, "retrained"); cells.append(f"{fl(r['objective']):.3f}" if r else "n.e.")
                rr = re_grp_get("case", "hyb", NR, s, t, t, NR, "designed"); cells.append(f"{fl(rr['objective']):.3f} $\\pm$ {fl(rr['opt_sd']):.3f}" if rr else "n.e.")
            lines.append(f"{CNAME[t]} & {s} & {sci(fl(o['k_on']))} & {fl(o['k_off']):.2f} & {fl(o['T']):.2f} & {1e6*fl(o['distance']):.1f} & {sci(fl(o['D']))} & {fl(o['N_max']):.0f} & {int(float(o['L']))} & {sci(fl(o['lam_best']), 0)} & {fl(o['koff_T']):.1f} & {fl(o['c_peak_over_KD']):.2g} & " + " & ".join(cells) + " \\\\")
    tab("TabCase", lines)
    # ---- wide domain
    for s in ("S1", "S4"):
        gains = []
        for t in TASKS:
            o = o_grp("wide", "det", 0, s, t); d = min(fl(r["objective"]) for r in det_rows(s, t))
            if o:
                g = -pct(fl(o["objective"]), d); gains.append(g)
                mac(f"Wide{SWORD[s]}{TWORD[t]}min", fl(o["objective"])); mac(f"Wide{SWORD[s]}{TWORD[t]}gainpct", g, "{:.0f}")
                for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("k_off", "Koff", "{:.2g}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}"), ("tau_D_over_T", "TauDT", "{:.2f}"), ("T", "T", "{:.2f}"), ("N_max", "Nmax", "{:.0f}")):
                    mac(f"Wide{SWORD[s]}{TWORD[t]}{name}", fl(o[k]), fmt)
                mac(f"Wide{SWORD[s]}{TWORD[t]}dum", 1e6 * fl(o["distance"]), "{:.1f}")
            oh = o_grp("wide", "hyb", 50000, s, t)
            if oh:
                r = re_grp_get("wide", "hyb", 50000, s, t, t, 50000, "designed")
                if r:
                    mac(f"WideNA{SWORD[s]}{TWORD[t]}NRhigh", fl(r["objective"]))
                    ref = get("hyb", 50000, s, t, "designed", 50000) if re_ else None
                    if ref: mac(f"WideNA{SWORD[s]}{TWORD[t]}NRhighgainpct", -pct(fl(r["objective"]), fl(ref["objective"])), "{:.0f}")
                    for k, name, fmt in (("koff_T", "KoffT", "{:.0f}"), ("k_off", "Koff", "{:.0f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}")):
                        mac(f"WideNA{SWORD[s]}{TWORD[t]}NRhigh{name}", fl(oh[k]), fmt)
        if gains:
            nz = lambda v: 0.0 if abs(v) < 0.5 else v
            mac(f"Wide{SWORD[s]}GainMin", nz(min(gains)), "{:.0f}"); mac(f"Wide{SWORD[s]}GainMax", nz(max(gains)), "{:.0f}"); mac(f"Wide{SWORD[s]}GainAbsMax", max(abs(g) for g in gains), "{:.0f}")
    wg = []
    for s in ("S1", "S4"):
        for t in TASKS:
            r = re_grp_get("wide", "hyb", 50000, s, t, t, 50000, "designed"); ref = get("hyb", 50000, s, t, "designed", 50000) if re_ else None
            if r and ref: wg.append(-pct(fl(r["objective"]), fl(ref["objective"])))
    if wg: mac("WideNAGainMin", (0.0 if abs(min(wg)) < 0.5 else min(wg)), "{:.0f}"); mac("WideNAGainMax", max(wg), "{:.0f}")
    lines = []
    for s in ("S1", "S4"):
        for t in TASKS:
            d = min(det_rows(s, t), key=lambda r: fl(r["objective"])); o = o_grp("wide", "det", 0, s, t)
            for label, r in (("default", d), ("wider", o)):
                if r is None: continue
                lines.append(f"{s} & {TNAME[t]} & {label} & {sci(fl(r['k_on']))} & {fl(r['k_off']):.2f} & {fl(r['T']):.2f} & {1e6*fl(r['distance']):.1f} & {sci(fl(r['D']))} & {fl(r['N_max']):.0f} & {int(float(r['L']))} & {fl(r['objective']):.4f} & {fl(r['koff_T']):.1f} & {fl(r['c_peak_over_KD']):.2g} & {fl(r['tau_D_over_T']):.2f} \\\\")
            oh = o_grp("wide", "hyb", 50000, s, t); rh = re_grp_get("wide", "hyb", 50000, s, t, t, 50000, "designed") if oh else None
            if oh and rh:
                lines.append(f"{s} & {TNAME[t]} & wider, $N_R=50000$ & {sci(fl(oh['k_on']))} & {fl(oh['k_off']):.2f} & {fl(oh['T']):.2f} & {1e6*fl(oh['distance']):.1f} & {sci(fl(oh['D']))} & {fl(oh['N_max']):.0f} & {int(float(oh['L']))} & {fl(rh['objective']):.4f} & {fl(oh['koff_T']):.1f} & {fl(oh['c_peak_over_KD']):.2g} & {fl(oh['tau_D_over_T']):.2f} \\\\")
    tab("TabWide", lines)
    # ---- compromise
    lines = []
    for s in ("S1", "S4"):
        for mode, NR in (("det", 0), ("hyb", 50000)):
            o = o_grp("compromise", mode, NR, s, "MG+SINE+MGCUBED")
            if not o: continue
            cells = []; worst = 0.0
            for t in TASKS:
                r = re_grp_get("compromise", mode, NR, s, "MG+SINE+MGCUBED", t, NR, "retrained")
                if mode == "det": ref = min(fl(x["objective"]) for x in det_rows(s, t))
                else:
                    rr = get("hyb", NR, s, t, "designed", NR) if re_ else None; ref = fl(rr["objective"]) if rr else float("nan")
                val = fl(r["objective"]) if r else float("nan"); ratio = val / ref if ref else float("nan"); worst = max(worst, ratio)
                cells.append(f"{val:.3f} ({ref:.3f})")
                mac(f"Comp{SWORD[s]}{TWORD[t]}{'MF' if mode == 'det' else 'NRhigh'}pct", 100 * (ratio - 1), "{:.0f}")
            mac(f"Comp{SWORD[s]}{'MF' if mode == 'det' else 'NRhigh'}worstpct", 100 * (worst - 1), "{:.0f}")
            for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}"), ("tau_D_over_T", "TauDT", "{:.2f}")):
                mac(f"Comp{SWORD[s]}{'MF' if mode == 'det' else 'NRhigh'}{name}", fl(o[k]), fmt)
            lamcell = " / ".join(sci(fl(o[f"lam_{tt}"]), 0) if o.get(f"lam_{tt}") else sci(fl(o["lam_best"]), 0) for tt in TASKS)
            lines.append(f"{s} & {'mean field' if mode == 'det' else f'$N_R={NR}$'} & {sci(fl(o['k_on']))} & {fl(o['k_off']):.2f} & {fl(o['T']):.2f} & {1e6*fl(o['distance']):.1f} & {sci(fl(o['D']))} & {fl(o['N_max']):.0f} & {int(float(o['L']))} & {lamcell} & {fl(o['koff_T']):.1f} & {fl(o['c_peak_over_KD']):.2g} & " + " & ".join(cells) + f" & {100*(worst-1):.0f} \\\\")
    tab("TabCompromise", lines)

# ---- switching
sw_path = GROUPS_DIR / "switching" / "switching_summary_NR50000.json"
if sw_path.exists():
    sw = json.load(open(sw_path)); PW = {"MG": "MG", "MGCUBED": "MGC", "GLUF": "Gluf", "GLUE": "Glue", "SINE": "Sine"}
    lines = []
    for r in sw:
        key = f"{PW[r['source']]}to{PW[r['target']]}"
        mac(f"Sw{key}LevelA", r["level_a_before"]); mac(f"Sw{key}LevelB", r["level_b_reference"]); mac(f"Sw{key}Tail", r["level_b_switched_tail"]); mac(f"Sw{key}Reuse", r["level_b_reuse"])
        mac(f"Sw{key}Reusepct", pct(r["level_b_reuse"], r["level_b_reference"]), "{:.0f}")
        for ramp in (0, 10, 50):
            mac(f"Sw{key}Peak{'abc'[(0,10,50).index(ramp)]}", r[f"peak_ramp{ramp}"], "{:.1f}")
            a = r[f"adapt_symbols_ramp{ramp}"]; mac(f"Sw{key}Adapt{'abc'[(0,10,50).index(ramp)]}", int(a) if a is not None else "none", "{:d}" if a is not None else "{}")
            asec = r[f"adapt_seconds_ramp{ramp}"]; mac(f"Sw{key}AdaptSec{'abc'[(0,10,50).index(ramp)]}", float(asec) if asec is not None else "none", "{:.0f}" if asec is not None else "{}")
        lines.append(f"{TNAME.get(r['source'], CNAME.get(r['source'], r['source']))} & {TNAME.get(r['target'], CNAME.get(r['target'], r['target']))} & {r['level_a_before']:.3f} & {r['level_b_reference']:.3f} & {r['level_b_reuse']:.3f} & " + " & ".join(f"{r[f'peak_ramp{x}']:.1f} / {r[f'adapt_symbols_ramp{x}'] if r[f'adapt_symbols_ramp{x}'] is not None else 'none'}" for x in (0, 10, 50)) + " \\\\")
    tab("TabSwitching", lines)
    for ramp, word in ((10, "Ten"), (50, "Fifty")):
        adr = [r[f"adapt_symbols_ramp{ramp}"] for r in sw if r[f"adapt_symbols_ramp{ramp}"] is not None]
        if adr: mac(f"SwAdapt{word}Min", min(adr), "{:d}"); mac(f"SwAdapt{word}Max", max(adr), "{:d}")
    for r in sw:
        if r["source"] in ("GLUF", "GLUE"):
            key = f"{PW[r['source']]}to{PW[r['target']]}"
            for ramp in (0, 10, 50):
                a = r[f"adapt_symbols_ramp{ramp}"]
                if a is not None: mac(f"Sw{key}AdaptMinutes{'abc'[(0,10,50).index(ramp)]}", 5 * int(a), "{:d}")
    mac("SwPairCount", len(set(frozenset((r["source"], r["target"])) for r in sw)), "{:d}"); mac("SwSwitchCount", len(sw), "{:d}")
    allad = [r[f"adapt_symbols_ramp{x}"] for r in sw for x in (0, 10, 50) if r[f"adapt_symbols_ramp{x}"] is not None]
    if allad: mac("SwAdaptMin", min(allad), "{:d}"); mac("SwAdaptMax", max(allad), "{:d}")
    ad0 = [r["adapt_symbols_ramp0"] for r in sw if r["adapt_symbols_ramp0"] is not None]
    if ad0: mac("SwAdaptZeroMin", min(ad0), "{:d}"); mac("SwAdaptZeroMax", max(ad0), "{:d}")
    mac("SwPeakZeroMax", max(r["peak_ramp0"] for r in sw), "{:.1f}"); mac("SwPeakFiftyMax", max(r["peak_ramp50"] for r in sw), "{:.1f}")
    mac("SwReusepctMin", min(pct(r["level_b_reuse"], r["level_b_reference"]) for r in sw), "{:.0f}"); mac("SwReusepctMax", max(pct(r["level_b_reuse"], r["level_b_reference"]) for r in sw), "{:.0f}")

# ---- capacity
cap = rd(GROUPS_DIR / "capacity.csv")
if cap:
    lines = []
    for cfg in ("nominal", "S1_MG", "S1_SINE", "S1_MGCUBED"):
        rows = sorted([r for r in cap if r["config"] == cfg], key=lambda r: float(r["N_R"]))
        name = "nominal" if cfg == "nominal" else f"S1 tuned for {TNAME[cfg[3:]]}"
        lines.append(f"{name} & " + " & ".join(f"{fl(r['linear_capacity']):.1f} / {fl(r['quadratic_capacity']):.1f}" for r in rows) + " \\\\")
        CW = {"nominal": "Nominal", "S1_MG": "SoneMG", "S1_SINE": "SoneSine", "S1_MGCUBED": "SoneMGC"}[cfg]
        for r in rows:
            nr = int(float(r["N_R"])); w = "MF" if nr == 0 else NRWORD.get(nr, f"N{nr}")
            mac(f"CapLin{CW}{w}", fl(r["linear_capacity"]), "{:.1f}"); mac(f"CapQuad{CW}{w}", fl(r["quadratic_capacity"]), "{:.1f}")
    tab("TabCapacity", lines)
    KW = {1: "Kone", 2: "Ktwo", 4: "Kfour", 5: "Kfive", 6: "Ksix", 8: "Keight"}
    for r in cap:
        if r["config"] != "S1_MG": continue
        nr = int(float(r["N_R"])); w = "MF" if nr == 0 else NRWORD.get(nr, f"N{nr}")
        for k, kw in KW.items(): mac(f"CapLin{kw}{w}", fl(r[f"lin_k{k}"]), "{:.2f}")
        mac(f"CapQuadKzero{w}", fl(r["quad_k0"]), "{:.2f}")

# ---- ranges of the noise-aware forecasting optima across receptor numbers (S1 operating point, S4 dissociation rate)
for s_, sw_ in (("S1", "Sone"), ("S4", "Sfour")):
    ro = [r for r in optima if r["mode"] == "hyb" and r["task"] == "MG" and r["set"] == s_]
    if ro:
        cp = [fl(r["c_peak_over_KD"]) for r in ro]; ko = [fl(r["k_off"]) for r in ro]
        mac(f"NA{sw_}MGCpeakKDmin", min(cp), "{:.2f}"); mac(f"NA{sw_}MGCpeakKDmax", max(cp), "{:.2f}")
        mac(f"NA{sw_}MGKoffMin", min(ko), "{:.2f}"); mac(f"NA{sw_}MGKoffMax", max(ko), "{:.2f}")

# ================================================================ additional tasks, readout designed for noise, joint searches, task map
ADD_DIR = mc.RESULTS / "additional_tasks_and_readouts"
BITS = ("XOR", "PAT", "NARMA"); BWORD = {"XOR": "Xor", "PAT": "Pat", "NARMA": "Narma"}; BNAME = {"XOR": "Temporal XOR", "PAT": "Bit-pattern detection", "NARMA": "NARMA-10"}
ALLW = dict(TWORD); ALLW.update(CWORD); ALLW.update(BWORD)
ALLN = {"MG": "MG forecasting", "SINE": "Sine-to-square", "MGCUBED": "MG-Cubed", "GLUF": "Biomarker forecast", "GLUE": "Threshold decision", "XOR": "Temporal XOR", "PAT": "Bit-pattern detection", "NARMA": "NARMA-10"}
NRW = dict(NRWORD); NRW.update({1_000_000: "NRmillion", 10_000_000: "NRtenmillion", 100_000_000: "NRhundredmillion", 1_000_000_000: "NRbillion"})
task_map_rows = rd(ADD_DIR / "task_map.csv"); crit_rows = rd(ADD_DIR / "critical_receptor_numbers.csv"); opt_add = rd(ADD_DIR / "optima.csv"); re_add = rd(ADD_DIR / "reevaluation.csv"); bt_add = rd(ADD_DIR / "baseline_taps.csv")
grid_rows = rd(ADD_DIR / "readout_grid.csv"); joint_opt = rd(ADD_DIR / "joint_readout_optima.csv")


def lget(task, s, proto, nr):
    for r in (task_map_rows or []):
        if r["task"] == task and r["set"] == s and r["protocol"] == proto and int(fl(r["N_R"])) == nr: return r
    return None


if task_map_rows:
    for t in ALLW:
        for s in ("S0", "S1", "S4"):
            for nr, w in NRW.items():
                r = lget(t, s, "retrained", nr)
                if r: mac(f"Ret{SWORD[s]}{ALLW[t]}{w}", fl(r["objective"]))
                r = lget(t, s, "remedy", nr)
                if r:
                    mac(f"Rem{SWORD[s]}{ALLW[t]}{w}", fl(r["objective"])); mac(f"Rem{SWORD[s]}{ALLW[t]}{w}held", fl(r["held"]))
                    mac(f"Rem{SWORD[s]}{ALLW[t]}{w}L", int(fl(r["L"])), "{:d}"); mac(f"Rem{SWORD[s]}{ALLW[t]}{w}Mp", int(fl(r["Mp"])), "{:d}"); mac(f"Rem{SWORD[s]}{ALLW[t]}{w}tau", fl(r["tau_over_T"]), "{:.2g}")
                r = lget(t, s, "joint", nr)
                if r: mac(f"Joint{SWORD[s]}{ALLW[t]}{w}", fl(r["objective"])); mac(f"Joint{SWORD[s]}{ALLW[t]}{w}L", int(fl(r["L"])), "{:d}"); mac(f"Joint{SWORD[s]}{ALLW[t]}{w}Mp", int(fl(r["Mp"])), "{:d}"); mac(f"Joint{SWORD[s]}{ALLW[t]}{w}tau", fl(r["tau_over_T"]), "{:.2g}")
    for t in ALLW:   # joint channel-plus-readout optima of S3 (S1 handled above)
        for nr, w in NRW.items():
            r = lget(t, "S3", "joint", nr)
            if r: mac(f"JointSthree{ALLW[t]}{w}", fl(r["objective"])); mac(f"JointSthree{ALLW[t]}{w}L", int(fl(r["L"])), "{:d}"); mac(f"JointSthree{ALLW[t]}{w}Mp", int(fl(r["Mp"])), "{:d}"); mac(f"JointSthree{ALLW[t]}{w}tau", fl(r["tau_over_T"]), "{:.2g}")
    for b in crit_rows:
        t = b["task"]
        for key, name in (("budget_retrained_S1", "BudgetRetSone"), ("budget_retrained_S4", "BudgetRetSfour"), ("budget_remedy_S1", "BudgetRemSone"), ("budget_remedy_S4", "BudgetRemSfour"),
                          ("level_retrained_S1", "LevelRetSone"), ("level_retrained_S4", "LevelRetSfour"), ("level_remedy_S1", "LevelRemSone"), ("level_remedy_S4", "LevelRemSfour")):
            v = b[key]
            if v == "none": mac(f"{name}{ALLW[t]}", "not reached for $N_R\\le10^{9}$")
            else:
                x = fl(v); e = int(math.floor(math.log10(x))); m = x / 10 ** e
                mac(f"{name}{ALLW[t]}", f"$\\le 500$" if x <= 500 else (f"${m:.1f}\\times10^{{{e}}}$" if e >= 4 else f"{round(x, -(e - 1)):.0f}"))
        if b["linear_ref"]: mac(f"LinRef{ALLW[t]}", fl(b["linear_ref"]))
    lines = []
    for b in crit_rows:
        t = b["task"]
        def fmt(v):
            if v == "none": return "n.r."
            x = fl(v)
            if x <= 500: return "$\\le 500$"
            e = int(math.floor(math.log10(x)))
            if x < 1e4: return f"{round(x, -(e - 1)):.0f}"
            return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"
        lines.append(f"{ALLN[t]} & {b['class']} & {fl(b['linear_ref']):.3f} & {fl(b['mf_S4']) if b['mf_S4'] else float('nan'):.3f} & {fmt(b['budget_retrained_S4'])} & {fmt(b['budget_remedy_S4'])} & {fmt(b['level_retrained_S4'])} & {fmt(b['level_remedy_S4'])} \\\\")
    tab("TabBudget", lines)
    for t in ALLW:
        for s in ("S1", "S4"):
            r = lget(t, s, "remedy_meanfield", 0)
            if r: mac(f"RemMF{SWORD[s]}{ALLW[t]}", fl(r["objective"])); mac(f"RemMF{SWORD[s]}{ALLW[t]}L", int(fl(r["L"])), "{:d}")
    # extended sweep table (retrained mean-field optima, all tasks, 500 to 1e9) for the SI
    NRALL = [500, 2000, 5000, 20000, 50000, 200000, 500000, 1000000, 10000000, 100000000, 1000000000]
    lines = []
    for t in ALLW:
        for s in ("S0", "S1", "S4"):
            cells = []
            for nr in NRALL:
                r = lget(t, s, "retrained", nr); cells.append(f"{fl(r['objective']):.3f}" if r else "n.e.")
            mf = lget(t, s, "meanfield", 0)
            if any(c != "n.e." for c in cells): lines.append(f"{ALLN[t]} & {s} & {fl(mf['objective']) if mf else float('nan'):.3f} & " + " & ".join(cells) + " \\\\")
    tab("TabNRext", lines)
# new tasks: mean-field optima and gains, fresh noise values
if opt_add:
    for t in BITS:
        z = [r for r in opt_add if r["mode"] == "det" and r["set"] == "S0" and r["task"] == t]
        for s in ("S0", "S1", "S4"):
            o = [r for r in opt_add if r["mode"] == "det" and r["set"] == s and r["task"] == t]
            if o:
                o = o[0]; mac(f"Bits{SWORD[s]}{BWORD[t]}min", fl(o["objective"])); mac(f"Bits{SWORD[s]}{BWORD[t]}held", fl(o["held"]))
                for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}"), ("tau_D_over_T", "TauDT", "{:.2f}"), ("occ_frac_above_0p8", "FracAbove", "{:.2f}")):
                    mac(f"Bits{SWORD[s]}{BWORD[t]}{name}", fl(o[k]), fmt)
                mac(f"Bits{SWORD[s]}{BWORD[t]}L", int(float(o["L"])), "{:d}")
                if z and s != "S0": mac(f"Bits{SWORD[s]}{BWORD[t]}gainpct", -pct(fl(o["objective"]), fl(z[0]["objective"])), "{:.0f}")
            for NR in (500, 50000):
                oh = [r for r in opt_add if r["mode"] == "hyb" and int(float(r["N_R"])) == NR and r["set"] == s and r["task"] == t]
                if oh:
                    rr = [r for r in (re_add or []) if r["design_mode"] == "hyb" and int(fl(r["design_NR"])) == NR and r["set"] == s and r["design_task"] == t and r["protocol"] == "designed"]
                    if rr: mac(f"BitsNA{SWORD[s]}{BWORD[t]}{NRWORD[NR]}", fl(rr[0]["objective"])); mac(f"BitsNA{SWORD[s]}{BWORD[t]}{NRWORD[NR]}sd", fl(rr[0]["opt_sd"]), "{:.3f}")
                    for k, name, fmt in (("koff_T", "KoffT", "{:.1f}"), ("c_peak_over_KD", "CpeakKD", "{:.2g}")):
                        mac(f"BitsNA{SWORD[s]}{BWORD[t]}{NRWORD[NR]}{name}", fl(oh[0][k]), fmt)
        if bt_add: mac(f"TapsBest{BWORD[t]}", min(fl(r["objective"]) for r in bt_add if r["task"] == t))
    lines = []
    for t in BITS:
        for s in ("S0", "S1", "S4"):
            o = [r for r in opt_add if r["mode"] == "det" and r["set"] == s and r["task"] == t]
            if not o: continue
            o = o[0]; cells = [f"{fl(o['objective']):.3f}"]
            for NR in (500, 50000):
                r = lget(t, s, "retrained", NR); cells.append(f"{fl(r['objective']):.3f}" if r else "n.e.")
                rr = [x for x in (re_add or []) if x["design_mode"] == "hyb" and int(fl(x["design_NR"])) == NR and x["set"] == s and x["design_task"] == t and x["protocol"] == "designed"]
                cells.append(f"{fl(rr[0]['objective']):.3f} $\\pm$ {fl(rr[0]['opt_sd']):.3f}" if rr else "n.e.")
            lines.append(f"{BNAME[t]} & {s} & {sci(fl(o['k_on']))} & {fl(o['k_off']):.2f} & {fl(o['T']):.2f} & {1e6*fl(o['distance']):.1f} & {sci(fl(o['D']))} & {fl(o['N_max']):.0f} & {int(float(o['L']))} & {sci(fl(o['lam_best']), 0)} & {fl(o['koff_T']):.1f} & {fl(o['c_peak_over_KD']):.2g} & " + " & ".join(cells) + " \\\\")
    tab("TabBits", lines)
    if bt_add: tab("TabBaselineTapsBits", [f"{BNAME[t]} & " + " & ".join(f"{fl(r['objective']):.3f}" for r in bt_add if r["task"] == t) + " \\\\" for t in BITS])
# summary tables of the readout grid (S1 and S4 optima): default readout, selected readout, its settings, held-out, per task and N_R
for SETREM in (("S1", "TabRemedies"), ("S4", "TabRemediesSfour")):
  if grid_rows:
    lines = []
    for t in ("MG", "SINE", "MGCUBED", "NARMA", "GLUF", "GLUE", "XOR", "PAT"):
        for NR in (500, 5000, 50000, 500000):
            rr = [r for r in grid_rows if r["task"] == t and r["set"] == SETREM[0] and int(fl(r["N_R"])) == NR]
            if not rr: continue
            base = [r for r in rr if r["variant"] == "sampled_default"][0]; best = min([r for r in rr if r["variant"] in ("integrating", "sampled_default")], key=lambda r: fl(r["objective"]))
            mac(f"RemDef{SWORD[SETREM[0]]}{ALLW[t]}{NRW[NR]}", fl(base["objective"]))
            lp = min([r for r in rr if r["variant"] == "integrating" and int(fl(r["L"])) == 5 and int(fl(r["Mp"])) == max(int(fl(x["Mp"])) for x in rr if x["variant"] == "integrating")], key=lambda r: fl(r["objective"]))
            lm = min([r for r in rr if r["variant"] == "integrating" and fl(r["tau_over_T"]) == 0.0 and int(fl(r["Mp"])) == max(int(fl(x["Mp"])) for x in rr if x["variant"] == "integrating")], key=lambda r: fl(r["objective"]))
            if SETREM[0] == "S1":
                mac(f"RemOnlyLP{ALLW[t]}{NRW[NR]}", fl(lp["objective"])); mac(f"RemOnlyL{ALLW[t]}{NRW[NR]}", fl(lm["objective"])); mac(f"RemOnlyLtau{ALLW[t]}{NRW[NR]}", fl(lp["tau_over_T"]), "{:.2g}"); mac(f"RemOnlyLL{ALLW[t]}{NRW[NR]}", int(fl(lm["L"])), "{:d}")
            setting = "default" if best["variant"] == "sampled_default" else f"{int(fl(best['Mp']))} / {fl(best['tau_over_T']):.2g} / {int(fl(best['L']))}"
            lines.append(f"{ALLN[t]} & {NR} & {fl(base['objective']):.3f} & {fl(lp['objective']):.3f} ({fl(lp['tau_over_T']):.2g}) & {fl(lm['objective']):.3f} ({int(fl(lm['L']))}) & {fl(best['objective']):.3f} & {setting} & {fl(best['held']):.3f} \\\\")
    tab(SETREM[1], lines)
# joint channel + readout optima table
joint_re = rd(ADD_DIR / "joint_readout_reevaluation.csv")
if joint_re:
    lines = []
    for r in sorted(joint_re, key=lambda r: (int(fl(r["N_R"])), r["set"], ["MG", "SINE", "MGCUBED", "NARMA"].index(r["task"]) if r["task"] in ["MG", "SINE", "MGCUBED", "NARMA"] else 9)):
        lines.append(f"{ALLN.get(r['task'], r['task'])} & {r['set']} & {int(fl(r['N_R']))} & {sci(fl(r['k_on']))} & {fl(r['k_off']):.2f} & {fl(r['T']):.2f} & {fl(r['N_max']):.0f} & {int(fl(r['Mp']))} & {fl(r['tau_over_T']):.2g} & {int(fl(r['L']))} & {sci(fl(r['lam']), 0)} & {fl(r['search_objective']):.3f} & {fl(r['objective']):.3f} $\\pm$ {fl(r['opt_sd']):.3f} & {fl(r['held']):.3f} & {fl(r['meanfield_same_readout']):.3f} \\\\")
        mac(f"JointMF{SWORD[r['set']]}{ALLW[r['task']]}{NRW[int(fl(r['N_R']))]}", fl(r["meanfield_same_readout"])); mac(f"JointSd{SWORD[r['set']]}{ALLW[r['task']]}{NRW[int(fl(r['N_R']))]}", fl(r["opt_sd"]), "{:.3f}")
    tab("TabJoint", lines); mac("NumJointSearches", len(rd(ADD_DIR / "joint_readout_searches.csv")), "{:d}")
elif joint_opt:
    lines = []
    for r in sorted(joint_opt, key=lambda r: (int(fl(r["N_R"])), r["set"], ["MG", "SINE", "MGCUBED", "NARMA"].index(r["task"]) if r["task"] in ["MG", "SINE", "MGCUBED", "NARMA"] else 9)):
        lines.append(f"{ALLN.get(r['task'], r['task'])} & {r['set']} & {int(fl(r['N_R']))} & {sci(fl(r['k_on']))} & {fl(r['k_off']):.2f} & {fl(r['T']):.2f} & {fl(r['N_max']):.0f} & {int(fl(r['Mp']))} & {fl(r['tau_over_T']):.2g} & {int(fl(r['L']))} & {fl(r['objective']):.3f} & {fl(r['held']):.3f} \\\\")
    tab("TabJoint", lines)

# ================================================================ cell-like readout and reference ESN
CELL_DIR = mc.RESULTS / "cell_like_readout_and_esn"
cell_rows = rd(CELL_DIR / "cell_like_readout.csv"); esn_rows = rd(CELL_DIR / "echo_state_network.csv")
NRWM = dict(NRW); NRWM[0] = "MF"


def _interp_budget(pts, level):
    """Receptor number at which the curve (sorted (N_R, value)) first falls below level, log-interpolated; None if never."""
    if not pts or level is None: return None
    if pts[0][1] < level: return pts[0][0]
    for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
        if e0 >= level > e1:
            return 10 ** (math.log10(n0) + (e0 - level) / (e0 - e1) * (math.log10(n1) - math.log10(n0)))
    return None


def _fmt_budget(x, cap="not reached for $N_R\\le5\\times10^{5}$"):
    if x is None: return cap
    e = int(math.floor(math.log10(max(x, 1.0))))
    if x <= 500 or (x < 1e4 and round(x, -(e - 1)) <= 500): return "$\\le 500$"
    if x < 1e4: return f"{round(x, -(e - 1)):.0f}"
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


# paired same-grid crossings: the sampled default of the readout-grid run (L of the optimum, ridge re-selected) against the
# selected readout (best of the integrating grid and that default) on the same realizations and receptor numbers
if grid_rows and crit_rows:
    lin_ref = {b["task"]: fl(b["linear_ref"]) for b in crit_rows}
    lines = []
    for t in ("MG", "SINE", "MGCUBED", "NARMA", "GLUF", "GLUE", "XOR", "PAT"):
        for s in ("S1", "S4"):
            dflt, best = [], []
            for NR in (500, 5000, 50000, 500000):
                rr = [r for r in grid_rows if r["task"] == t and r["set"] == s and int(fl(r["N_R"])) == NR]
                if not rr: continue
                dflt.append((NR, min(fl(r["objective"]) for r in rr if r["variant"] == "sampled_default")))
                best.append((NR, min(fl(r["objective"]) for r in rr if r["variant"] in ("integrating", "sampled_default"))))
            if not dflt: continue
            cells = []
            for level, word in ((lin_ref.get(t), "Ref"), (0.25, "Level")):
                for pts, proto in ((dflt, "Def"), (best, "Rem")):
                    x = _interp_budget(pts, level)
                    mac(f"Paired{word}{proto}{SWORD[s]}{ALLW[t]}", _fmt_budget(x)); cells.append(_fmt_budget(x, "n.r."))
            lines.append(f"{ALLN[t]} & {s} & " + " & ".join(cells) + " \\\\")
    tab("TabPairedBudget", lines)


if cell_rows:
    refs = {b["task"]: fl(b["linear_ref"]) for b in (crit_rows or [])}
    two_gain = []
    for t in ALLW:
        for s in ("S1", "S4"):
            curve1, curve12 = [], []
            for nr in (0, 500, 5000, 50000, 500000):
                rr = [r for r in cell_rows if r["task"] == t and r["set"] == s and int(fl(r["N_R"])) == nr]
                if not rr: continue
                w = NRWM[nr]
                one = min([r for r in rr if r["variant"] == "integrating" and int(fl(r["Mp"])) == 1], key=lambda r: fl(r["objective"]))
                both = min([r for r in rr if r["variant"] == "integrating" and int(fl(r["Mp"])) in (1, 2)], key=lambda r: fl(r["objective"]))
                base = [r for r in rr if r["variant"] == "sampled_default"][0]
                mac(f"CellOne{SWORD[s]}{ALLW[t]}{w}", fl(one["objective"])); mac(f"CellOne{SWORD[s]}{ALLW[t]}{w}held", fl(one["held"]))
                mac(f"CellOne{SWORD[s]}{ALLW[t]}{w}tau", fl(one["tau_over_T"]), "{:.2g}"); mac(f"CellOne{SWORD[s]}{ALLW[t]}{w}L", int(fl(one["L"])), "{:d}")
                two = min([r for r in rr if r["variant"] == "integrating" and int(fl(r["Mp"])) == 2], key=lambda r: fl(r["objective"]))
                mac(f"CellTwo{SWORD[s]}{ALLW[t]}{w}", fl(two["objective"])); mac(f"CellTwo{SWORD[s]}{ALLW[t]}{w}held", fl(two["held"]))
                mac(f"CellTwo{SWORD[s]}{ALLW[t]}{w}tau", fl(two["tau_over_T"]), "{:.2g}"); mac(f"CellTwo{SWORD[s]}{ALLW[t]}{w}L", int(fl(two["L"])), "{:d}")
                mac(f"Cell{SWORD[s]}{ALLW[t]}{w}", fl(both["objective"])); mac(f"Cell{SWORD[s]}{ALLW[t]}{w}Mp", int(fl(both["Mp"])), "{:d}")
                mac(f"CellDef{SWORD[s]}{ALLW[t]}{w}", fl(base["objective"]))
                if nr > 0:
                    curve1.append((nr, fl(one["objective"]))); curve12.append((nr, fl(two["objective"])))
                    two_gain.append(100.0 * (fl(one["objective"]) - fl(two["objective"])) / fl(one["objective"]))
            mac(f"CellBudget{SWORD[s]}{ALLW[t]}", _fmt_budget(_interp_budget(sorted(curve1), refs.get(t))))
            mac(f"CellLevel{SWORD[s]}{ALLW[t]}", _fmt_budget(_interp_budget(sorted(curve1), 0.25)))
            mac(f"CellBudgetTwo{SWORD[s]}{ALLW[t]}", _fmt_budget(_interp_budget(sorted(curve12), refs.get(t))))
            mac(f"CellLevelTwo{SWORD[s]}{ALLW[t]}", _fmt_budget(_interp_budget(sorted(curve12), 0.25)))
    if two_gain: mac("CellTwoGainMaxpct", max(two_gain), "{:.0f}"); mac("CellTwoGainMinpct", min(two_gain), "{:.0f}")
    # both tunable-parameter sets (S1 and S4) are tabulated, since the main text quotes S4 two-window values
    lines = []
    for t in ALLW:
        for s in ("S1", "S4"):
            rr = [r for r in cell_rows if r["task"] == t and r["set"] == s]
            if not rr: continue
            mf = lget(t, s, "meanfield", 0)
            for mp, word in ((1, ""), (2, "Two")):
                cells = []
                for nr in (0, 500, 5000, 50000, 500000):
                    sub = [r for r in rr if int(fl(r["N_R"])) == nr and r["variant"] == "integrating" and int(fl(r["Mp"])) == mp]
                    cells.append(f"{min(fl(r['objective']) for r in sub):.3f}" if sub else "n.e.")
                short = lambda v: "n.r." if str(v).startswith("not reached") else v
                lines.append(f"{ALLN[t] if (mp == 1 and s == 'S1') else ''} & {s if mp == 1 else ''} & {mp} & {refs.get(t, float('nan')):.3f} & {fl(mf['objective']) if mf else float('nan'):.3f} & " + " & ".join(cells) + f" & {short(numbers.get('CellBudget' + word + SWORD[s] + ALLW[t], ''))} & {short(numbers.get('CellLevel' + word + SWORD[s] + ALLW[t], ''))} \\\\")
    tab("TabCellLike", lines)
    lines = []
    for t in ALLW:
        for s in ("S1", "S4"):
            for mp in (1, 2):
                for nr in (500, 5000, 50000, 500000):
                    sub = [r for r in cell_rows if r["task"] == t and r["set"] == s and int(fl(r["N_R"])) == nr and r["variant"] == "integrating" and int(fl(r["Mp"])) == mp]
                    if not sub: continue
                    b = min(sub, key=lambda r: fl(r["objective"]))
                    lines.append(f"{ALLN[t]} & {s} & {mp} & {nr} & {fl(b['objective']):.3f} $\\pm$ {fl(b['opt_sd']):.3f} & {fl(b['tau_over_T']):.2g} & {int(fl(b['L']))} & {sci(fl(b['lam']), 0)} & {fl(b['held']):.3f} \\\\")
    tab("TabCellLikeSettings", lines)
if esn_rows:
    refs = {b["task"]: fl(b["linear_ref"]) for b in (crit_rows or [])}
    lines = []; beats4 = 0; ntask = 0; esn_means = {}
    for t in ALLW:
        rr = [r for r in esn_rows if r["task"] == t]
        if not rr: continue
        b = min(rr, key=lambda r: fl(r["search_objective"]))
        mac(f"Esn{ALLW[t]}", small3(fl(b["fresh_mean"]))); mac(f"EsnSd{ALLW[t]}", fl(b["fresh_sd"]), "{:.3f}"); mac(f"EsnHeld{ALLW[t]}", small3(fl(b["held_mean"]))); esn_means[t] = fl(b["fresh_mean"])
        mac(f"EsnSearch{ALLW[t]}", fl(b["search_objective"])); mac(f"EsnN{ALLW[t]}", int(fl(b["N"])), "{:d}")
        mac(f"EsnRho{ALLW[t]}", fl(b["rho"]), "{:.2f}"); mac(f"EsnSin{ALLW[t]}", fl(b["s_in"]), "{:.2g}"); mac(f"EsnLeak{ALLW[t]}", fl(b["leak"]), "{:.2f}"); mac(f"EsnSb{ALLW[t]}", fl(b["s_b"]), "{:.2f}")
        mf1 = lget(t, "S1", "meanfield", 0); mf4 = lget(t, "S4", "meanfield", 0)
        if mf4:
            mac(f"EsnOverSfour{ALLW[t]}", fl(b["fresh_mean"]) / fl(mf4["objective"]), "{:.2f}"); ntask += 1; beats4 += fl(b["fresh_mean"]) < fl(mf4["objective"])
            if fl(b["fresh_mean"]) > 1e-3: mac(f"EsnFactor{ALLW[t]}", fl(mf4["objective"]) / fl(b["fresh_mean"]), "{:.0f}" if fl(mf4["objective"]) / fl(b["fresh_mean"]) >= 10 else "{:.1f}")
        esncell = "${<}0.001$" if fl(b["fresh_mean"]) < 5e-4 else f"{fl(b['fresh_mean']):.3f} $\\pm$ {fl(b['fresh_sd']):.3f}"
        lines.append(f"{ALLN[t]} & {int(fl(b['N']))} & {esncell} & {small3(fl(b['held_mean']))} & {fl(mf1['objective']) if mf1 else float('nan'):.3f} & {fl(mf4['objective']) if mf4 else float('nan'):.3f} & {refs.get(t, float('nan')):.3f} \\\\")
    mac("EsnBeatsSfourCount", beats4, "{:d}"); mac("EsnTaskCount", ntask, "{:d}"); mac("NumEsnSearches", len(esn_rows), "{:d}")
    if esn_means.get("XOR", 1.0) < 1e-3 and esn_means.get("PAT", 1.0) < 1e-3: mac("EsnBitsCeil", "0.001")
    tab("TabESN", lines)

# ---- scaling of the model geometry (Discussion): a receiver alpha times larger at alpha times the distance with a
# symbol interval alpha^2 times longer has the same dimensionless groups; alpha from the case-study sampling interval
T_CASE, T_NOM, R_RX, D_NOM = 300.0, 1.25, 3e-6, 7.07e-6
alpha = math.sqrt(T_CASE / T_NOM)
mac("ScaleAlphaCase", alpha, "{:.1f}"); mac("ScaleAlphaSqCase", T_CASE / T_NOM, "{:.0f}")
mac("ScaleRadiusCase", R_RX * alpha * 1e6, "{:.0f}"); mac("ScaleDistanceCase", D_NOM * alpha * 1e6, "{:.0f}")
area_cm2 = 4 * math.pi * (R_RX * alpha * 100) ** 2
mac("ScaleReceptorsCaseLow", sci(area_cm2 * 1e11, 1)); mac("ScaleReceptorsCaseHigh", sci(area_cm2 * 1e12, 1))
cell_cm2 = 4 * math.pi * (R_RX * 100) ** 2
mac("CellReceptorsLow", sci(cell_cm2 * 1e11, 1)); mac("CellReceptorsHigh", sci(cell_cm2 * 1e12, 1))

# ---------------------------------------------------------------- write
(OUT / "numbers.tex").write_text("% generated by make_numbers.py, do not edit\n" + "\n".join(macros) + "\n")
_sdnote = re.compile(r"(\S+) \$\\pm\$ 0\.000(?=[\s&\\])")
tables = {k: _sdnote.sub(r"\1$^{*}$", v) for k, v in tables.items()}   # a sample SD below 0.0005 is marked with an asterisk, defined in the table caption
for _k, _v in tables.items():
    if "$^{*}$" in _v: print("asterisk note needed in the caption of", _k)
(OUT / "tables.tex").write_text("% generated by make_numbers.py, do not edit\n" + "\n".join(f"\\newcommand{{\\{k}}}{{%\n{v}\n}}" for k, v in tables.items()) + "\n")
print(len(macros), "macros,", len(tables), "tables ->", OUT)
