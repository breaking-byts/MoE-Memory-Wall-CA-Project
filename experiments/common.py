"""Shared experiment settings and helpers."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from prefetch import costs, predictors as P  # noqa: E402
from prefetch.workload import Workload  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results")
FIGURES = os.path.join(ROOT, "experiments", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# real-router workloads first; the synthetic ones bracket them
WORKLOADS = ["tinymoe-prose", "tinymoe-code", "mixtral-calibrated", "uniform"]
STEPS = 192                         # decode steps simulated per sequence (tokens 64..255)
BATCH_GROUPS = {1: 8, 4: 8, 8: 4}   # batch size -> number of independent batches (seq groups)


def groups(batch, n_groups):
    return [list(range(g * batch, (g + 1) * batch)) for g in range(n_groups)]


def kv_act_gib(batch, ctx, model=costs.MIXTRAL):
    kv = 2 * model.layers * 8 * 128 * model.bytes_per * batch * ctx   # 8 KV heads x 128 dims
    return kv / costs.GiB + 1.0                                         # + activations / workspace


def capacities(batch, plat=costs.H100_NVL5):
    """HBM expert-cache sizes (experts; per-layer partitions -> multiples of 32).
    The largest is everything that fits in 80 GB next to the non-expert weights,
    KV cache and activations (224 of 256 experts = 73.5 GiB)."""
    cmax = costs.max_cache_experts(plat, kv_act_gib=kv_act_gib(batch, 512 + STEPS))
    return [32, 64, 128, 192, cmax // 32 * 32]


class PredictorSet:
    """Count tables + tuned mixing weights for one workload (profiling split only)."""

    def __init__(self, wl):
        self.wl = wl
        self.tab = P.Tables(wl.profile_ids)
        self.lam = P.tune_lambda(self.tab, wl.profile_ids)

    def make(self, name):
        t, lam = self.tab, self.lam
        return {"random": lambda: P.RandomPred(),
                "frequency": lambda: P.Frequency(t),
                "temporal": lambda: P.Temporal(t),
                "spatial": lambda: P.Spatial(t),
                "st-hybrid": lambda: P.STHybrid(t, lam),
                "st-moe": lambda: P.STMoE(t),
                "spec-gate": lambda: P.SpecGate(getattr(self.wl, "calib", None)),
                "spec-hybrid": lambda: P.SpecHybrid(t, lam, calib=getattr(self.wl, "calib", None)),
                "oracle": lambda: P.Oracle()}[name]()

    def available(self):
        names = ["random", "frequency", "temporal", "spatial", "st-hybrid", "st-moe"]
        if self.wl.spec:
            names += ["spec-gate", "spec-hybrid"]
        return names


def fmt_budget(b):
    if b[0] == "fixed":
        return f"top-{b[1]}"
    if b[0] == "adaptive":
        return f"adaptive(tau={b[1]},max={b[2]})"
    if b[0] == "min_p":
        return "ST-MoE score>=2 (variable set)"
    return "variable set"


def write_csv(rows, path):
    import csv
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", os.path.relpath(path, ROOT))
