"""F1: accuracy of the training-free expert predictors against the routed experts.

Part A  every workload x predictor x lookahead x prefetch budget, batch 1 and 4
        -> results/f1_accuracy.csv, results/f1_recall_per_layer.csv
Part B  sensitivity of the Mixtral-calibrated workload to the unknown cross-layer
        strength beta -> results/f1_beta_sweep.csv
Figures -> figures/f1_accuracy.png
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from prefetch.evaluate import evaluate  # noqa: E402

N_SEQ, N_STEPS = 16, 128
MAIN = {"st-hybrid", "spec-gate", "spec-hybrid"}
BUDGETS_MAIN = [("fixed", 2), ("fixed", 3), ("fixed", 4),
                ("adaptive", 0.8, 6), ("adaptive", 0.9, 6), ("adaptive", 0.95, 6)]
BUDGETS_OTHER = [("fixed", 2), ("adaptive", 0.9, 6)]


def part_a():
    rows, per_layer = [], []
    for wname in WORKLOADS:
        wl = Workload(wname)
        ps = PredictorSet(wl)
        for batch in (1, 4):
            for pname in ps.available():
                ds = sorted(wl.spec) if pname.startswith("spec") else [1, 2]   # spec0 only for TinyMoE
                if pname in ("frequency", "temporal", "random"):
                    ds = [1]                       # these ignore the lookahead
                if pname == "st-moe":
                    ds = [1]                       # defined for the next layer only
                for d in ds:
                    budgets = [("min_p", 0.25)] if pname == "st-moe" else (
                        BUDGETS_MAIN if pname in MAIN else BUDGETS_OTHER)
                    for bud in budgets:
                        t = time.time()
                        r = evaluate(ps.make(pname), wl, d, bud, slots=range(N_SEQ),
                                     batch=batch, max_steps=N_STEPS)
                        rows.append({"workload": wname, "batch": batch, "predictor": pname,
                                     "lookahead": d, "budget": fmt_budget(bud),
                                     "recall": r["recall"], "precision": r["precision"],
                                     "full_coverage": r["coverage"], "prefetch_set": r["size"],
                                     "n_predictions": r["n"]})
                        if bud == ("fixed", 2) or pname == "st-moe":
                            per_layer.append({"workload": wname, "batch": batch, "predictor": pname,
                                              "lookahead": d, "budget": fmt_budget(bud),
                                              **{f"L{l}": v for l, v in enumerate(r["recall_per_layer"])}})
                        print(f"{wname:18s} B{batch} {pname:14s} d{d} {fmt_budget(bud):24s} "
                              f"recall {r['recall']:.3f} prec {r['precision']:.3f} "
                              f"cov {r['full_coverage'] if 'full_coverage' in r else r['coverage']:.3f} "
                              f"|P| {r['size']:.2f}  ({time.time() - t:.1f}s)", flush=True)
    write_csv(rows, os.path.join(RESULTS, "f1_accuracy.csv"))
    write_csv(per_layer, os.path.join(RESULTS, "f1_recall_per_layer.csv"))
    return rows


class _WL:
    """In-memory workload wrapper for the beta sweep."""

    def __init__(self, z):
        self.ids = z["ids"].astype(np.int64)
        self.profile_ids = z["profile_ids"].astype(np.int64)
        self.prompt_len = int(z["prompt_len"])
        self.spec = {d: z[f"spec{d}"].astype(np.float32) / 255 for d in (1, 2)}
        self.calib = {d: z[f"calib{d}"] for d in (1, 2)}
        self.S, self.T, self.L, self.K = self.ids.shape
        self.E = 8

    @property
    def decode_steps(self):
        return self.T - self.prompt_len


def part_b():
    sys.path.insert(0, os.path.join(ROOT, "workloads"))
    import synthetic
    rows = []
    for beta in (0.0, 0.3, 0.55, 0.7, 0.85):
        z = synthetic.build_mixtral_calibrated(beta)
        wl = _WL(z)
        calib_err = float(np.max(np.abs(z["fit_first_or_second"] - z["target_first_or_second"])[[0, 15, 31]]))
        ps = PredictorSet(wl)
        for pname in ("temporal", "spatial", "st-hybrid", "st-moe", "spec-gate"):
            bud = ("min_p", 0.25) if pname == "st-moe" else ("fixed", 2)
            r = evaluate(ps.make(pname), wl, 1, bud, slots=range(N_SEQ), max_steps=N_STEPS)
            rows.append({"beta": beta, "table5_max_abs_error": calib_err, "predictor": pname,
                         "budget": fmt_budget(bud),
                         "recall": r["recall"], "precision": r["precision"],
                         "full_coverage": r["coverage"], "prefetch_set": r["size"]})
            print(f"beta {beta:.2f} {pname:12s} recall {r['recall']:.3f}", flush=True)
    write_csv(rows, os.path.join(RESULTS, "f1_beta_sweep.csv"))


def plot():
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "f1_accuracy.csv"))))
    pl = list(csv.DictReader(open(os.path.join(RESULTS, "f1_recall_per_layer.csv"))))
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    colors = {"temporal": "#7f7f7f", "spatial": "#bcbd22", "st-hybrid": "#1f77b4", "st-moe": "#9467bd",
              "spec-gate": "#ff7f0e", "spec-hybrid": "#d62728", "frequency": "#8c564b", "random": "#cccccc"}
    for i, w in enumerate(WORKLOADS):
        a = ax[0, i]
        for p in colors:
            pts = [(float(r["prefetch_set"]), float(r["recall"])) for r in rows
                   if r["workload"] == w and r["predictor"] == p and r["batch"] == "1"
                   and r["lookahead"] == ("1" if p != "spec-gate" else "1")]
            if pts:
                pts.sort()
                a.plot(*zip(*pts), "o-", color=colors[p], label=p, ms=4)
        a.axhline(0.25, ls=":", c="k", lw=0.8)
        a.set_title(f"{w}: recall vs prefetch budget (B=1, d=1)")
        a.set_xlabel("experts prefetched per layer (|P|)")
        a.set_ylabel("recall = |P∩A|/|A|")
        a.set_ylim(0, 1.02)
        a.legend(fontsize=7)
        b = ax[1, i]
        for r in pl:
            if r["workload"] == w and r["batch"] == "1" and r["lookahead"] == "1" and r["predictor"] in colors:
                l0 = 1 if r["predictor"] == "spec-gate" else 0          # undefined for layer 0
                b.plot(range(l0, 32), [float(r[f"L{l}"]) for l in range(l0, 32)],
                       color=colors[r["predictor"]], label=r["predictor"])
        b.axhline(0.25, ls=":", c="k", lw=0.8)
        b.set_title(f"{w}: per-layer recall (top-2 / ST-MoE set)")
        b.set_xlabel("target layer")
        b.set_ylim(0, 1.02)
    fig.savefig(os.path.join(FIGURES, "f1_accuracy.png"), dpi=110)
    print("wrote experiments/figures/f1_accuracy.png")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        part_a()
        part_b()
    plot()
