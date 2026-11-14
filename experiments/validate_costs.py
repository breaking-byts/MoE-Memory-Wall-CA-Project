"""Validation of the modified simulator and of the timeline's cost model.

For every simulator run under experiments/sim_out/ (created by f2_prefetch.py):
  V1  the routing the simulator executed (routing_log) == the routing trace given to it
  V2  per step, the timeline cost model reproduces the simulator's expert_ffn column
      in compute mode (expert execution) and copy mode (expert loading)
  V3  per step, bytes implied by the simulator's moe_offload energy == routed experts x 352 MB
Plus a regression check of the midterm D2 numbers with the modified simulator
(random routing, no trace): 68.13 ms load / 25.02 ms exec.
-> results/validate_costs.csv ; exits non-zero on any failure.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from prefetch.simulator import OUT, load  # noqa: E402
from prefetch.workload import read_sim_log  # noqa: E402

PLAT = {"H100-NVL5": costs.H100_NVL5, "A100-NVL4": costs.A100_NVL4}


def check_run(tag):
    sim = load(tag)
    plat = PLAT[sim["platform"]]
    trace = os.path.join(OUT, tag, "routing_trace.txt")
    with open(trace) as f:
        B, steps, L, K = map(int, f.readline().split())
        arr = np.array(f.read().split(), dtype=np.int64).reshape(B, steps, L, K)
    R = arr.transpose(1, 2, 0, 3)
    v1 = bool((sim["R_compute"] == R).all() and (sim["R_copy"] == R).all())
    e_exec, e_load, e_bytes = [], [], []
    for i, (rc, ry) in enumerate(zip(sim["rows_compute"], sim["rows_copy"])):
        ex = ld = nb = 0.0
        for l in range(L):
            ids, cnt = np.unique(R[i, l].ravel(), return_counts=True)
            act = sum(3 * m * costs.MIXTRAL.inter * 2 / plat.exec_bw for m in cnt)
            ex += sum(costs.expert_exec_s(m, plat) for m in cnt) + costs.gate_s(B, plat)
            ld += len(ids) * costs.expert_load_s(plat) + act + costs.MIXTRAL.hidden * 8 * 2 / plat.link_bw
            nb += len(ids) * costs.MIXTRAL.expert_bytes + costs.MIXTRAL.hidden * 8 * 2   # + router weights
        e_exec.append(rc["expert_ffn"] * 1e-9 / ex - 1)
        e_load.append(ry["expert_ffn"] * 1e-9 / ld - 1)
        sim_bytes = ry["moe_offload"] * 1e-9 / costs.copy_energy_j(1, plat)
        e_bytes.append(sim_bytes / nb - 1)
    return {"run": tag, "steps": len(e_exec), "routing_log_matches_trace": v1,
            "max_rel_err_exec": float(np.max(np.abs(e_exec))),
            "max_rel_err_load": float(np.max(np.abs(e_load))),
            "max_rel_err_offload_bytes": float(np.max(np.abs(e_bytes)))}


def midterm_regression():
    """Midterm D2 configs re-run with the modified simulator (run_final.sh step 2)
    must reproduce the committed midterm CSVs (ssd_results/)."""
    import csv
    out = {"run": "midterm D2 configs on the modified simulator", "steps": 1}
    worst = 0.0
    for mode in ("GPU_compute", "offloaded_weight_copy"):
        new = next(csv.DictReader(open(glob.glob(os.path.join(OUT, "midterm_regression", f"*{mode}.csv"))[0])))
        old = next(csv.DictReader(open(glob.glob(os.path.join(ROOT, "ssd_results", f"*{mode}.csv"))[0])))
        for k, v in old.items():
            try:
                a, b = float(v), float(new[k])
            except ValueError:
                continue
            if a != 0:
                worst = max(worst, abs(b / a - 1))
        out["midterm_load_ms" if mode != "GPU_compute" else "midterm_exec_ms"] = float(new["expert_ffn"]) * 1e-6
    out["max_rel_diff_vs_committed_midterm_csv"] = worst
    return out


if __name__ == "__main__":
    tags = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(OUT, "*", "done.json")))
    tags = [t for t in tags if not t.startswith("test") and t != "midterm_regression"]
    out = [check_run(t) for t in tags]
    ok = all(r["routing_log_matches_trace"] and r["max_rel_err_exec"] < 1e-9 and
             r["max_rel_err_load"] < 1e-9 and r["max_rel_err_offload_bytes"] < 1e-5 for r in out)
    m = midterm_regression()
    ok &= m["max_rel_diff_vs_committed_midterm_csv"] < 1e-6
    out.append(m)
    write_csv(out, os.path.join(RESULTS, "validate_costs.csv"))
    worst = max(max(r.get("max_rel_err_exec", 0), r.get("max_rel_err_load", 0)) for r in out)
    print(f"{len(tags)} simulated batches ({2 * len(tags)} simulator runs) checked; worst relative latency error {worst:.2e}; "
          f"offload-bytes error < 1e-5 (float32 energy constant); midterm D2 regression "
          f"{m['midterm_load_ms']:.2f} / {m['midterm_exec_ms']:.2f} ms -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
