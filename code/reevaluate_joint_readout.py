"""Fresh re-evaluation (R realizations, new seeds) of the joint channel-plus-readout optima at their design N_R.
Usage: python reevaluate_joint_readout.py <R> <n_workers>
Input: joint_readout_optima.csv; output: joint_readout_reevaluation.csv (both in
data/results/additional_tasks_and_readouts). The manuscript used R = 50."""
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor
import readout_design as m
mc = m.mc; RES = mc.RESULTS / "additional_tasks_and_readouts"; R = int(sys.argv[1]); NW = int(sys.argv[2])


def fl(x):
    try: return float(x)
    except Exception: return float("nan")


def job(a):
    k, o = a
    p = mc.params_from_row(o); NR = int(fl(o["N_R"]))
    r = m.evaluate_readout(o["task"], p, int(fl(o["Mp"])), fl(o["tau_over_T"]), int(fl(o["L"])), mode="hyb", N_R=NR, R=R, seed=90000 + 3 * k)
    d = m.evaluate_readout(o["task"], p, int(fl(o["Mp"])), fl(o["tau_over_T"]), int(fl(o["L"])), mode="det")
    return dict(task=o["task"], set=o["set"], N_R=NR, R=R, Mp=int(fl(o["Mp"])), tau_over_T=fl(o["tau_over_T"]), L=int(fl(o["L"])), k_on=fl(o["k_on"]), k_off=fl(o["k_off"]), T=fl(o["T"]), N_max=fl(o["N_max"]),
                lam=r["lam"], objective=r["objective"], opt_sd=r["opt_sd"], held=r["held"], search_objective=fl(o["objective"]), meanfield_same_readout=d["objective"])


if __name__ == "__main__":
    optima = list(csv.DictReader(open(RES / "joint_readout_optima.csv", encoding="utf-8-sig")))
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for r in ex.map(job, list(enumerate(optima))):
            rows.append(r); print(f"{r['set']} {r['task']:8s} NR={r['N_R']:6d} fresh {r['objective']:.4f} (search {r['search_objective']:.4f}, mean field same readout {r['meanfield_same_readout']:.4f}) Mp {r['Mp']} tau/T {r['tau_over_T']:.2f} L {r['L']} {time.time()-t0:.0f} s", flush=True)
    with open(RES / "joint_readout_reevaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote joint_readout_reevaluation.csv")
