"""F2-F4: prefetching vs on-demand execution.

For every workload, batch size and independent sequence group:
  1. run the modified SSD-offloading simulator on the group's routing trace
     (compute + copy modes) -> per-step attention latency, energies, and the
     routing the simulator actually executed;
  2. replay that routing through the two-stream timeline for every policy and
     HBM expert-cache capacity.
Every finished run is appended to results/f2_runs.jsonl, so the script can be
interrupted and resumed.  Aggregates -> results/f2_summary.csv.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from prefetch.simulator import run_sim, per_step_inputs, serial_on_demand_ms  # noqa: E402
from prefetch.timeline import Sim, Policy, summarize  # noqa: E402

RUNS = os.path.join(RESULTS, "f2_runs.jsonl")
F2_GROUPS = {1: 16, 4: 8, 8: 4}    # B=4 and B=8 use all 32 evaluation sequences
SIM_PLATFORMS = {"H100-NVL5": costs.H100_NVL5, "A100-NVL4": costs.A100_NVL4}


OURS = dict(budget=("fixed", 2), guided_replacement=True, preempt=True, feedback=True)
APEX = dict(OURS, budget=("adaptive", 0.9, 4))


def policies(ps, cap):
    """(label, Policy) pairs.  Prefetch policies use lookahead d=1 (issued when the
    previous layer has been routed, ST-MoE timing) unless stated otherwise.
    'ours' = predictor + top-2 per token + prediction-guided replacement +
    preemptible prefetch + accuracy-feedback throttling.  '+ adaptive budget' adds
    the APEX-style top-(k+delta) rule (tau=0.9, <=4 experts per token)."""
    mk = ps.make
    out = [
        ("on-demand, no reuse (framework copy volume)", Policy("od-noreuse", capacity=cap, reuse=False)),
        ("on-demand + LRU", Policy("od-lru", capacity=cap)),
        ("prefetch ST-MoE (re-impl.)",                   # its score ranks residents for replacement
         Policy("pf", cap, True, mk("st-moe"), 1, ("min_p", 0.25), guided_replacement=True)),
        ("prefetch ST-MoE predictor + our mechanisms",   # separates predictor from mechanism
         Policy("pf", cap, True, mk("st-moe"), 1, ("min_p", 0.25), guided_replacement=True, preempt=True,
                feedback=True)),
        ("prefetch ours: ST-hybrid", Policy("pf", cap, True, mk("st-hybrid"), 1, **OURS)),
    ]
    if ps.wl.spec:
        out += [("prefetch ours: spec-hybrid", Policy("pf", cap, True, mk("spec-hybrid"), 1, **OURS)),
                ("prefetch ours: spec-hybrid + adaptive budget", Policy("pf", cap, True, mk("spec-hybrid"), 1, **APEX))]
    out += [("oracle prefetch (upper bound)",
             Policy("oracle", cap, True, mk("oracle"), 1, ("set",), guided_replacement=True, preempt=True))]
    return out


def ablation_policies(ps, cap):
    """Step-by-step build-up of the prefetcher (one change per row, A1..A4 cumulative)."""
    best = "spec-hybrid" if ps.wl.spec else "st-hybrid"
    mk = ps.make
    A = [("A0 on-demand + LRU", Policy("od-lru", capacity=cap)),
         ("A1 top-2 prefetch, LRU replacement", Policy("pf", cap, True, mk(best), 1, ("fixed", 2))),
         ("A2 + prediction-guided replacement", Policy("pf", cap, True, mk(best), 1, ("fixed", 2), guided_replacement=True)),
         ("A3 + preemptible prefetch", Policy("pf", cap, True, mk(best), 1, ("fixed", 2), guided_replacement=True, preempt=True)),
         ("A4 + accuracy-feedback throttle (= ours)", Policy("pf", cap, True, mk(best), 1, **OURS)),
         ("A5 ours + adaptive budget (APEX-style)", Policy("pf", cap, True, mk(best), 1, **APEX)),
         ("A6 ours + adaptive budget + link-slack gate", Policy("pf", cap, True, mk(best), 1, slack_aware=True, **APEX)),
         ("A7 ours, static confidence floor 0.3 instead of feedback",
          Policy("pf", cap, True, mk(best), 1, ("fixed", 2), guided_replacement=True, preempt=True, p_min=0.3)),
         ("A8 ours, lookahead d=2", Policy("pf", cap, True, mk(best), 2, **OURS)),
         ("A9 ours with history-only predictor (ST-hybrid)", Policy("pf", cap, True, mk("st-hybrid"), 1, **OURS))]
    if 0 in ps.wl.spec:                                   # needs pre-attention router outputs (TinyMoE)
        A.append(("A10 ours, pre-attention issue d=0 (APEX timing)", Policy("pf", cap, True, mk("spec-hybrid"), 0, **OURS)))
    A.append(("oracle prefetch (upper bound)",
              Policy("oracle", cap, True, mk("oracle"), 1, ("set",), guided_replacement=True, preempt=True)))
    return A


def done_keys():
    if not os.path.exists(RUNS):
        return set()
    return {json.loads(l)["key"] for l in open(RUNS)}


def main(platforms=("H100-NVL5",), workloads=WORKLOADS, batches=(1, 4, 8), caps=None, labels=None,
         policy_fn=policies, tag="main", n_groups=None):
    done = done_keys()
    for pname in platforms:
        plat = SIM_PLATFORMS[pname]
        for wname in workloads:
            wl = Workload(wname)
            ps = PredictorSet(wl)
            for B in batches:
                for gi, slots in enumerate(groups(B, n_groups or F2_GROUPS[B])):
                    H, R = wl.batch(slots, STEPS)
                    sim = run_sim(R, f"{pname}_{wname}_B{B}_g{gi}", platform=pname)
                    assert (sim["R_compute"] == R).all() and (sim["R_copy"] == R).all(), \
                        "simulator routed different experts than the trace"
                    nonexp, efix = per_step_inputs(sim, plat)
                    serial = serial_on_demand_ms(sim)
                    Rs = R[:len(nonexp)]                       # steps with a CSV row
                    spec = {d: v[slots] for d, v in wl.spec.items()}
                    for cap in (caps or capacities(B, plat)):
                        for label, pol in policy_fn(ps, cap):
                            if labels and label not in labels:
                                continue
                            key = f"{tag}|{pname}|{wname}|B{B}|g{gi}|C{cap}|{label}"
                            if key in done:
                                continue
                            t = time.time()
                            per_step, st = Sim(Rs, H, wl.prompt_len, nonexp, plat, pol, spec).run()
                            s = summarize(per_step, st, plat, costs.MIXTRAL, efix)
                            rec = {"key": key, "experiment": tag, "platform": pname, "workload": wname, "batch": B,
                                   "group": gi, "capacity": cap, "policy": label, **s,
                                   "sim_serial_on_demand_ms_per_step": serial,
                                   "sim_expert_exec_ms_per_step": float(np.mean([r["expert_ffn"] for r in sim["rows_compute"]])) * 1e-6,
                                   "sim_expert_load_ms_per_step": float(np.mean([r["expert_ffn"] for r in sim["rows_copy"]])) * 1e-6}
                            with open(RUNS, "a") as f:
                                f.write(json.dumps(rec) + "\n")
                            print(f"{key:80s} lat {s['latency_ms_per_step']:7.2f} ms  stall "
                                  f"{s['stall_ms_per_step']:6.2f}  {time.time() - t:.1f}s", flush=True)


def summary():
    rows = [json.loads(l) for l in open(RUNS)]
    agg = {}
    for r in rows:
        k = (r["experiment"], r["platform"], r["workload"], r["batch"], r["capacity"], r["policy"])
        agg.setdefault(k, []).append(r)
    out = []
    num = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("batch", "group", "capacity")]
    for k, rs in sorted(agg.items()):
        row = dict(zip(("experiment", "platform", "workload", "batch", "capacity", "policy"), k))
        row["groups"] = len(rs)
        for n in num:
            row[n] = float(np.mean([r[n] for r in rs]))
        for n in ("latency_ms_per_step", "stall_ms_per_step", "energy_total_j_per_step"):
            row[n + "_std"] = float(np.std([r[n] for r in rs], ddof=1)) if len(rs) > 1 else 0.0
        row["latency_ms_per_token"] = row["latency_ms_per_step"]          # one token per sequence per step
        # expert-load latency = time the DRAM->HBM link spends copying experts per step
        # (the part not hidden behind computation is stall_ms_per_step)
        row["expert_load_ms_per_step"] = row["traffic_gb_per_step"] * 1e9 / SIM_PLATFORMS[k[1]].link_bw * 1e3
        row["energy_j_per_token"] = row["energy_total_j_per_step"] / k[3]
        out.append(row)
    # speed-up / savings vs the on-demand + LRU baseline at the same point
    BASE = {"main": "on-demand + LRU", "ablation": "A0 on-demand + LRU"}   # same groups as the row
    base = {(r["experiment"], r["platform"], r["workload"], r["batch"], r["capacity"]): r for r in out
            if r["policy"] == BASE.get(r["experiment"])}
    for r in out:
        b = base.get((r["experiment"], r["platform"], r["workload"], r["batch"], r["capacity"]))
        if b:
            r["speedup_vs_od_lru"] = b["latency_ms_per_step"] / r["latency_ms_per_step"]
            r["stall_reduction_vs_od_lru"] = 1 - r["stall_ms_per_step"] / max(b["stall_ms_per_step"], 1e-12)
            r["energy_saving_vs_od_lru"] = 1 - r["energy_total_j_per_step"] / b["energy_total_j_per_step"]
            r["extra_traffic_vs_od_lru"] = r["traffic_gb_per_step"] / max(b["traffic_gb_per_step"], 1e-12) - 1
    # per-group speed-up spread (each group compared with its own baseline run)
    per = {(r["experiment"], r["platform"], r["workload"], r["batch"], r["group"], r["capacity"], r["policy"]): r
           for r in rows}
    spread = {}
    for (e, p, w, b, g, c, pol), r in per.items():
        bl = per.get((e, p, w, b, g, c, BASE.get(e)))
        if bl:
            spread.setdefault((e, p, w, b, c, pol), []).append(bl["latency_ms_per_step"] / r["latency_ms_per_step"])
    for row in out:
        sp = spread.get((row["experiment"], row["platform"], row["workload"], row["batch"], row["capacity"],
                         row["policy"]), [])
        if len(sp) > 1:
            row["speedup_min_over_groups"], row["speedup_max_over_groups"] = float(min(sp)), float(max(sp))
    write_csv(out, os.path.join(RESULTS, "f2_summary.csv"))
    import gzip
    import shutil as _sh
    # compact, byte-reproducible copy that is committed (no file name / timestamp in the gzip header)
    with open(RUNS, "rb") as fi, open(RUNS + ".gz", "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fo:
        _sh.copyfileobj(fi, fo)


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        main()
        # ablation of the prefetcher design (batch 1, three cache sizes)
        main(workloads=["tinymoe-prose", "mixtral-calibrated"], batches=(1,), caps=[64, 128, 224],
             policy_fn=ablation_policies, tag="ablation", n_groups=8)
        # A100-class 80 GB HBM (the project's reference local memory), NVLink 4.0 450 GB/s
        main(platforms=("A100-NVL4",), batches=(1,), caps=[64, 128, 224], tag="main", n_groups=8)
    summary()
