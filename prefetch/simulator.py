"""Drive the (modified) SSD-offloading LLMSimulator on a routing trace.

For one decode batch it runs the simulator twice -- collects_gpu_compute on
(expert execution) and off (expert loading) -- with the batch's routing given
by `simulation: routing_trace`, and reads back
  * the per-step CSV rows (latency + energy breakdown), and
  * the routing log (= the experts the simulator actually routed).
Results are cached under experiments/sim_out/<tag>/ so plotting scripts can
re-use them without re-running the simulator.
"""
import csv
import glob
import json
import os
import shutil
import subprocess

import numpy as np
import yaml

from .workload import ROOT, write_sim_trace, read_sim_log

SIM_DIR = os.path.join(ROOT, "SSD-offloading", "LLMSimulator", "build")
OUT = os.path.join(ROOT, "experiments", "sim_out")
BASE_CFG = os.path.join(ROOT, "configs", "final_base.yaml")

GPU = {"H100-NVL5": ("H100", 5), "H100-NVL4": ("H100", 4), "A100-NVL4": ("A100", 4)}


def run_sim(R, tag, platform="H100-NVL5", input_len=512, force=False):
    """R [steps, L, B, K] -> dict(rows_compute, rows_copy, routed R_sim)."""
    out = os.path.join(OUT, tag)
    done = os.path.join(out, "done.json")
    if os.path.exists(done) and not force:
        return load(tag)
    os.makedirs(out, exist_ok=True)
    steps, L, B, K = R.shape
    gpu, nvl = GPU[platform]
    write_sim_trace(R, os.path.join(out, "routing_trace.txt"))
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["system"]["gpu_gen"] = gpu
    cfg["system"]["offload"]["offload_bandwidth"] = nvl
    cfg["serving"]["max_batch_size"] = B
    cfg["simulation"]["input_len"] = input_len
    cfg["simulation"]["output_len"] = steps + 2
    cfg["simulation"]["iter"] = steps            # iter n -> n-1 CSV rows (rows = steps 0..n-2)
    # Each mode runs in its own working directory (the framework reads ./config.yaml and
    # ./dram_config_*.yaml from its cwd), so several batches can be simulated in parallel.
    # All paths in the config are relative to that directory.
    cfg["simulation"]["routing_trace"] = "../routing_trace.txt"
    for mode, collect in (("compute", True), ("copy", False)):
        work = os.path.join(out, f"work_{mode}")
        os.makedirs(work, exist_ok=True)
        os.makedirs(os.path.join(out, f"csv_{mode}"), exist_ok=True)
        for f in glob.glob(os.path.join(SIM_DIR, "dram_config*.yaml")):
            shutil.copy(f, work)
        cfg["system"]["offload"]["collects_gpu_compute"] = collect
        cfg["simulation"]["routing_log"] = f"../routing_log_{mode}.csv"
        cfg["log"]["output_directory"] = f"../csv_{mode}"
        with open(os.path.join(work, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(os.path.join(out, f"sim_{mode}.log"), "w") as log:
            subprocess.run([os.path.join(SIM_DIR, "run")], cwd=work, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        for f in glob.glob(os.path.join(work, "dram_config*.yaml")):   # inputs only: do not keep copies
            os.remove(f)
        shutil.rmtree(os.path.join(work, "log"), ignore_errors=True)
    json.dump({"steps": steps, "B": B, "platform": platform}, open(done, "w"))
    return load(tag)


def _rows(path):
    f = glob.glob(os.path.join(path, "*.csv"))
    assert len(f) == 1, f
    return [{k: (float(v) if k not in ("type", "##Latency", "##Energy") else v) for k, v in r.items()}
            for r in csv.DictReader(open(f[0]))]


def load(tag):
    out = os.path.join(OUT, tag)
    meta = json.load(open(os.path.join(out, "done.json")))
    L, K = 32, 2
    rc, ry = _rows(os.path.join(out, "csv_compute")), _rows(os.path.join(out, "csv_copy"))
    Rc = read_sim_log(os.path.join(out, "routing_log_compute.csv"), L, K)
    Ry = read_sim_log(os.path.join(out, "routing_log_copy.csv"), L, K)
    return {"rows_compute": rc, "rows_copy": ry, "R_compute": Rc, "R_copy": Ry, **meta}


def serial_on_demand_ms(sim):
    """The framework's own on-demand estimate without any overlap: per step,
    attention block + expert loading (copy mode) + expert execution (compute mode)."""
    rc, ry = sim["rows_compute"], sim["rows_copy"]
    return float(np.mean([(a["MoE_decoders"] + b["expert_ffn"]) * 1e-6 for a, b in zip(rc, ry)]))


def per_step_inputs(sim, plat, L=32):
    """Policy-independent per-step quantities taken from the simulator rows.

    nonexp: GPU time per layer before the experts can start = attention block
    (QKV, attention, O-proj, norms) from the CSV + the router GEMV, which the
    CSV books inside expert_ffn."""
    from . import costs
    rc = sim["rows_compute"]
    g = costs.gate_s(sim["B"], plat)
    nonexp = np.array([(r["MoE_decoders"] - r["expert_ffn"]) * 1e-9 / L + g for r in rc])
    e_fixed = np.array([(r["fc_dram"] + r["fc_comp"] + r["attn_dram"] + r["attn_comp"]
                         + r["moe_hbm"] + r["moe_comp"]) * 1e-9 for r in rc])
    return nonexp, e_fixed
