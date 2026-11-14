"""F3/F4 figures and headline table from results/f2_summary.csv.

figures/f2_latency_capacity.png   latency per token vs HBM expert-cache capacity
figures/f2_breakdown.png          headline configs: latency split, traffic split, energy split
results/f3_headline.csv           headline comparison table (also printed)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAIN = ["on-demand, no reuse (framework copy volume)", "on-demand + LRU", "prefetch ST-MoE (re-impl.)",
        "prefetch ST-MoE predictor + our mechanisms", "prefetch ours: ST-hybrid", "prefetch ours: spec-hybrid",
        "prefetch ours: spec-hybrid + adaptive budget", "oracle prefetch (upper bound)"]
COL = dict(zip(MAIN, ["#7f7f7f", "#000000", "#9467bd", "#c5b0d5", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]))
SHORT = {"on-demand, no reuse (framework copy volume)": "on-demand\nno reuse", "on-demand + LRU": "on-demand\n+LRU",
         "prefetch ST-MoE (re-impl.)": "ST-MoE\n(re-impl.)",
         "prefetch ST-MoE predictor + our mechanisms": "ST-MoE pred.\n+our mech.",
         "prefetch ours: ST-hybrid": "ours\nST-hybrid",
         "prefetch ours: spec-hybrid": "ours\nspec-hybrid", "prefetch ours: spec-hybrid + adaptive budget": "ours spec\n+adaptive",
         "oracle prefetch (upper bound)": "oracle"}


def load():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "f2_summary.csv"))))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fig_capacity(rows):
    batches = sorted({r["batch"] for r in pick(rows, experiment="main", platform="H100-NVL5")})
    fig, ax = plt.subplots(len(batches), len(WORKLOADS), figsize=(5 * len(WORKLOADS), 3.8 * len(batches)),
                           constrained_layout=True, squeeze=False)
    for i, B in enumerate(batches):
        for j, w in enumerate(WORKLOADS):
            a = ax[i, j]
            for p in MAIN:
                pts = sorted((r["capacity"], r["latency_ms_per_token"]) for r in
                             pick(rows, experiment="main", platform="H100-NVL5", workload=w, batch=B, policy=p))
                if pts:
                    a.plot(*zip(*pts), "o-", c=COL[p], label=p, ms=4)
            a.set_title(f"{w}, batch {int(B)} (H100, 900 GB/s link)")
            a.set_xlabel("HBM expert-cache capacity (experts; 256 = whole model)")
            a.set_ylabel("latency per decode step (ms)")
            a.set_ylim(bottom=0)
            if i == 0 and j == 0:
                a.legend(fontsize=7)
    fig.savefig(os.path.join(FIGURES, "f2_latency_capacity.png"), dpi=100)


def fig_breakdown(rows, configs):
    fig, ax = plt.subplots(len(configs), 3, figsize=(19, 4.3 * len(configs)), constrained_layout=True, squeeze=False)
    for i, cfg in enumerate(configs):
        sel = [r for r in pick(rows, **cfg)]
        sel.sort(key=lambda r: list(SHORT).index(r["policy"]) if r["policy"] in SHORT else 99)
        x = np.arange(len(sel))
        lab = [SHORT.get(r["policy"], r["policy"]) for r in sel]
        busy = [r["latency_ms_per_step"] - r["stall_ms_per_step"] for r in sel]
        stall = [r["stall_ms_per_step"] for r in sel]
        a = ax[i, 0]
        a.bar(x, busy, label="GPU busy (attention + experts)", color="#4c72b0")
        a.bar(x, stall, bottom=busy, label="exposed expert-load stall", color="#dd8452")
        a.set_xticks(x, lab, fontsize=7); a.set_ylabel("ms per decode step")
        a.set_title(f"latency: {cfg['workload']}, B={int(cfg['batch'])}, C={int(cfg['capacity'])}, {cfg['platform']}")
        a.legend(fontsize=7)
        a = ax[i, 1]
        gb = costs.MIXTRAL.expert_bytes / 1e9 / r_steps(sel)
        dem = [r["demand_fetches"] * gb[k] for k, r in enumerate(sel)]
        use = [(r["prefetch_fetches"] - r["prefetch_wasted"]) * gb[k] for k, r in enumerate(sel)]
        was = [r["prefetch_wasted"] * gb[k] for k, r in enumerate(sel)]
        a.bar(x, dem, label="demand fetches", color="#55a868")
        a.bar(x, use, bottom=dem, label="prefetches used", color="#8172b2")
        a.bar(x, was, bottom=np.add(dem, use), label="prefetches wasted (full copies)", color="#c44e52")
        abo = [r["aborted_gb_per_step"] for r in sel]
        a.bar(x, abo, bottom=np.add(np.add(dem, use), was), label="aborted partial copies", color="#ccb974")
        a.set_xticks(x, lab, fontsize=7); a.set_ylabel("GB moved DRAM->HBM per step")
        a.set_title("off-chip memory traffic"); a.legend(fontsize=7)
        a = ax[i, 2]
        fx = [r["energy_fixed_j_per_step"] for r in sel]
        cp = [r["energy_copy_j_per_step"] for r in sel]
        bg = [r["energy_background_j_per_step"] - r["energy_stall_background_j_per_step"] for r in sel]
        sb = [r["energy_stall_background_j_per_step"] for r in sel]
        a.bar(x, fx, label="compute + HBM (ops)", color="#4c72b0")
        a.bar(x, cp, bottom=fx, label="DRAM->HBM copies", color="#55a868")
        b2 = np.add(fx, cp)
        a.bar(x, bg, bottom=b2, label="background power x busy time", color="#937860")
        a.bar(x, sb, bottom=np.add(b2, bg), label="background power x stall time", color="#dd8452")
        a.set_xticks(x, lab, fontsize=7); a.set_ylabel("J per decode step")
        a.set_title("energy"); a.legend(fontsize=7)
    fig.savefig(os.path.join(FIGURES, "f2_breakdown.png"), dpi=100)


def r_steps(sel):
    return np.array([r["steps"] for r in sel])


def headline(rows, configs):
    out = []
    for cfg in configs:
        for r in pick(rows, **cfg):
            out.append({**cfg, "policy": r["policy"],
                        "latency_ms_per_token": r["latency_ms_per_token"],
                        "expert_load_ms_per_step": r["expert_load_ms_per_step"],
                        "expert_load_stall_ms": r["stall_ms_per_step"],
                        "stall_share": r["stall_frac"],
                        "hit_rate": r["hit_rate"], "miss_rate": r["miss_rate"],
                        "prefetch_accuracy": r["prefetch_accuracy"],
                        "prefetch_eventual_use": r["prefetch_eventual_use"],
                        "traffic_gb_per_step": r["traffic_gb_per_step"],
                        "energy_j_per_token": r["energy_j_per_token"],
                        "speedup_vs_od_lru": r.get("speedup_vs_od_lru", ""),
                        "stall_reduction_vs_od_lru": r.get("stall_reduction_vs_od_lru", ""),
                        "energy_saving_vs_od_lru": r.get("energy_saving_vs_od_lru", ""),
                        "extra_traffic_vs_od_lru": r.get("extra_traffic_vs_od_lru", "")})
    write_csv(out, os.path.join(RESULTS, "f3_headline.csv"))
    for r in out:
        print(f"{r['workload']:16s} B{int(r['batch'])} C{int(r['capacity']):3d} {r['policy']:38s} "
              f"{r['latency_ms_per_token']:7.2f} ms  stall {r['expert_load_stall_ms']:6.2f}  "
              f"{r['traffic_gb_per_step']:5.1f} GB  {r['energy_j_per_token']:6.2f} J/tok")


if __name__ == "__main__":
    rows = load()
    configs = [dict(experiment="main", platform="H100-NVL5", workload=w, batch=b, capacity=c)
               for w, b, c in (("tinymoe-prose", 1.0, 64.0), ("tinymoe-prose", 1.0, 224.0),
                               ("mixtral-calibrated", 1.0, 64.0), ("mixtral-calibrated", 1.0, 224.0),
                               ("tinymoe-prose", 4.0, 128.0))]
    configs = [c for c in configs if pick(rows, **c)]
    fig_capacity(rows)
    fig_breakdown(rows, configs)
    headline(rows, configs)
    print("wrote experiments/figures/f2_latency_capacity.png, f2_breakdown.png")
