"""D2: single GPU, all Mixtral experts in off-chip DRAM -> expert load vs execution time.

Inputs (produced by the runs listed in the README / explainer page):
  run_mixtral_single.log            LLMSimulator, 1 x H100, all weights in HBM
  ssd_results/*GPU_compute.csv      SSD-offloading cpu.yaml      (execution time)
  ssd_results/*weight_copy.csv      SSD-offloading cpu_copy.yaml (loading time)
"""
import csv
import glob
import re

HBM_CAP_GIB = 80.0                      # H100 memory_capacity (hardware_config.h)
HBM_BW = 3.352e12                       # H100 HBM3, B/s
LINK_BW = 900e9                         # offload_bandwidth: 5 -> NVLink-C2C 5.0, B/s
PEAK_FLOPS = 989.4e12                   # H100 FP16
CPU_TO_HBM_PJ_BIT = 9.2 + 4.2           # DDR5 read + HBM write (module_graph.cpp)
HBM_READ_PJ_BIT = 4.2
TOKENS, TOP_K, LAYERS = 4, 2, 32        # cpu*.yaml: batch 4, one decode step
EXPERT_PARAMS = 3 * 4096 * 14336        # gate/up/down projections of one expert


def row(pattern):
    return next(csv.DictReader(open(glob.glob(pattern)[0])))


# 1. Without off-chip memory the model does not fit on one GPU.
log = open("run_mixtral_single.log").read()
weights = float(re.search(r"Weight: ([\d.]+)GB", log).group(1))
total = float(re.search(r"Total: ([\d.]+)GB", log).group(1))
print(f"Single H100: weights {weights:.2f} GiB, total {total:.2f} GiB "
      f"> {HBM_CAP_GIB:.0f} GiB -> OOM. Experts must live off-chip.\n")

# 2. Load vs execution, experts in CPU memory.
exe = row("ssd_results/*CPU_memory*GPU_compute.csv")
load = row("ssd_results/*CPU_memory*weight_copy.csv")
exec_ms = float(exe["expert_ffn"]) / 1e6
load_ms = float(load["expert_ffn"]) / 1e6
print(f"expert loading time   (cpu_copy.yaml) = {load_ms:8.2f} ms")
print(f"expert execution time (cpu.yaml)      = {exec_ms:8.2f} ms")
print(f"loading / execution                   = {load_ms / exec_ms:8.2f} x\n")

# 3. Same numbers from first principles.
moved = float(load["moe_offload"]) * 1e-9 / (CPU_TO_HBM_PJ_BIT * 1e-12 * 8)  # bytes
flops = 2 * TOKENS * TOP_K * LAYERS * EXPERT_PARAMS
print(f"routed expert bytes moved = {moved / 1e9:.2f} GB "
      f"({moved / (EXPERT_PARAMS * 2):.0f} expert loads of {EXPERT_PARAMS * 2 / 1e6:.0f} MB)")
print(f"  load = bytes / link BW            = {moved / LINK_BW * 1e3:8.2f} ms")
print(f"  exec = bytes / (HBM BW - link BW) = {moved / (HBM_BW - LINK_BW) * 1e3:8.2f} ms"
      "  (framework subtracts the copy from HBM BW)")
print(f"  pure arithmetic = FLOPs / peak    = {flops / PEAK_FLOPS * 1e3:8.3f} ms"
      "  -> execution itself is HBM-bandwidth bound\n")

# 4. Energy per byte moved.
e_load, e_exec = float(load["moe_offload"]) / 1e9, float(exe["moe_hbm"]) / 1e9
print(f"energy: CPU->HBM copy {e_load:.2f} J vs HBM read {e_exec:.2f} J = "
      f"{e_load / e_exec:.2f} x  (= {CPU_TO_HBM_PJ_BIT:.1f}/{HBM_READ_PJ_BIT} pJ/bit)")
