#!/usr/bin/env python3

import os
import sys
import subprocess
import yaml
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE  = os.path.join(BASE_DIR, "build", "config.yaml")
SIM_CMD      = [os.path.join(BASE_DIR, "build", "run")]

# latency
LATENCY_NON_MOE_DECODER   = "non_MoE_decoders"
LATENCY_FC_COLS = [
    "qkvgen", "q_down_proj", "kv_down_proj", "kr_proj",
    "q_up_proj", "qr_proj", "kv_up_proj", "tr_k_up_proj",
    "v_up_proj", "o_proj", "ffn", "shared_expert_ffn"
]
LATENCY_ATTEN = "atten_gen"
LATENCY_MOE   = "expert_ffn"
LATENCY_COMM  = "communication"

# energy
BASE = {
    "fc_dram":         "fc_dram",
    "fc_comp":         "fc_comp",
    "ffn_dram":        "ffn_dram",
    "ffn_comp":        "ffn_comp",
    "shared_moe_hbm": "shared_moe_hbm",
    "shared_moe_comp": "shared_moe_comp",
    "attn_dram":       "attn_dram",
    "attn_comp":       "attn_comp",
    "moe_hbm":        "moe_hbm",
    "moe_comp":        "moe_comp",
}
OFFLOAD = "moe_offload"

# model & batch size
EXPERIMENTS = {
    "mixtral":    [1, 4, 16, 64, 256, 1024],
    "deepseekR1": [4, 16, 64, 256, 1024],
}

# system background power (Watt)
## 70: single H100 SXM 80GB GPU (includes HBM)
## 57: 256GB of CPU memory (DDR5-7200 x8) per GPU
## ignore the negligible background power of SSD (https://semiconductor.samsung.com/consumer-storage/internal-ssd/9100-pro/)
BACKGROUND_POWER = 57 + 70

DISCARD_COLS = [
    "non_MoE_decoders", "MoE_decoders", "qkvgen", "q_down_proj", "kv_down_proj", "kr_proj",
    "q_up_proj", "qr_proj", "kv_up_proj", "tr_k_up_proj",
    "v_up_proj", "atten_sum", "atten_gen", "o_proj",
    "ffn", "shared_expert_ffn", "communication", "fc_dram",
    "fc_comp", "attn_dram", "attn_comp", "ffn_dram", "ffn_comp",
    "shared_moe_hbm", "shared_moe_comp", "moe_hbm", "moe_comp"
]

def load_cfg():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_cfg(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

def update_configuration(model_name: str, batch_size: int, input_sequence_length: int):
    cfg = load_cfg()
    cfg['model']['model_name'] = model_name
    cfg['serving']['max_batch_size'] = batch_size
    cfg['simulation']['input_len'] = input_sequence_length
    cfg['simulation']['output_len'] = 2 # the first decode stage after the prefill stage
    cfg['simulation']['iter'] = 2

    if model_name == "mixtral":
        cfg['system']['num_node'] = 1
        cfg['system']['num_device'] = 4
        cfg['system']['distribution']['expert_tensor_degree'] = 1
        cfg['system']['distribution']['none_expert_tensor_degree'] = 4
        cfg['system']['optimization']['compressed_kv'] = False
        cfg['system']['optimization']['use_absorb'] = False
    elif model_name == "deepseekR1":
        cfg['system']['num_node'] = 4
        cfg['system']['num_device'] = 8
        cfg['system']['distribution']['expert_tensor_degree'] = 1
        cfg['system']['distribution']['none_expert_tensor_degree'] = 8
        cfg['system']['optimization']['compressed_kv'] = True
        cfg['system']['optimization']['use_absorb'] = True

    cfg['system']['offload']['ssd_energy_scale'] = 1.0
    cfg['system']['offload']['offload_expert_to_cpu_memory'] = False
    cfg['system']['offload']['offload_expert_to_ssd']       = False
    write_cfg(cfg)


def run_sim(target_gpu: bool, offload_cpu: bool, offload_ssd: bool, nvlink_gen: int = 4) -> str:
    cfg = load_cfg()
    cfg['system']['offload']['offload_expert_to_cpu_memory'] = offload_cpu
    cfg['system']['offload']['offload_expert_to_ssd']       = offload_ssd
    cfg['system']['offload']['collects_gpu_compute']       = target_gpu
    cfg['system']['offload']['offload_bandwidth']       = nvlink_gen
    write_cfg(cfg)

    tag = "CPU_memory" if offload_cpu else "SSD" if offload_ssd else "baseline"
    collect = "GPU_compute" if target_gpu else "offloaded_weight_copy"
    print(f"\n--- Running [{tag}] ---")
    simulation_working_dir = "./build"
    proc = subprocess.run(SIM_CMD, cwd=simulation_working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"ERROR: failed ({tag})")

    # file name
    m   = cfg['model']['model_name']
    d   = cfg['simulation']['data']
    nn  = cfg['system']['num_node']
    nd  = cfg['system']['num_device']
    ne_tp_dg = cfg["system"]["distribution"]["none_expert_tensor_degree"]
    ne_dp_dg = int(nd * nn / ne_tp_dg)
    bs  = cfg['serving']['max_batch_size']
    od  = cfg['log']['output_directory']
    fname = os.path.join(simulation_working_dir, od, f"{m}_{d}_N{nn}_D{nd}_TP{ne_tp_dg}_DP{ne_dp_dg}_batchsize{bs}_{tag}_NVLink{nvlink_gen}_{collect}.csv")
    if not os.path.isfile(fname):
        sys.exit(f"ERROR: cannot find generated CSV: {fname}")
    print(f"→ OK: {fname}")

    if not target_gpu:
        zero_out_columns_in_csv(fname)
    
    return fname

def zero_out_columns_in_csv(fname: str):
    df = pd.read_csv(fname)
    for col in DISCARD_COLS:
        if col in df.columns:
            df[col] = 0
    df.to_csv(fname, index=False)

def compute_energy_baseline(label: str, latency: float, df_base: pd.DataFrame, bs: int):
    cfg = load_cfg()
    num_tot_device = cfg['system']['num_node'] * cfg['system']['num_device']

    r = df_base.loc[0]
    # xx_off: (memory/storage access) off-chip memory access
    # xx_on: (GPU-compute) on-chip memory & compute
    fc_off   = (r[BASE["fc_dram"]] + r[BASE["ffn_dram"]] + r[BASE["shared_moe_hbm"]]) / 1e9
    fc_on    = (r[BASE["fc_comp"]] + r[BASE["ffn_comp"]] + r[BASE["shared_moe_comp"]]) / 1e9
    attn_off = r[BASE["attn_dram"]] / 1e9
    attn_on  = r[BASE["attn_comp"]] / 1e9
    moe_off  = r[BASE["moe_hbm"]] / 1e9
    moe_on   = r[BASE["moe_comp"]] / 1e9
    background   = num_tot_device * BACKGROUND_POWER * latency / 1e3
    per_tok  = (fc_off + fc_on + attn_off + attn_on + moe_off + moe_on + background) / bs

    print(f"\n>> {label} energy (J/single decode stage)")
    print(f"   FC off-chip      : {fc_off:.6f}")
    print(f"   FC on-chip       : {fc_on:.6f}")
    print(f"   Attn off-chip    : {attn_off:.6f}")
    print(f"   Attn on-chip     : {attn_on:.6f}")
    print(f"   MoE off-chip     : {moe_off:.6f}")
    print(f"   MoE on-chip      : {moe_on:.6f}")
    print(f"   System background: {background:.6f}")
    print(f"   ---------------------------")
    print(f"   Per token        : {per_tok:.6f} (÷ batch_size={bs})")

def compute_energy_offload(label: str, latency: float, df_gpu: pd.DataFrame, df_offload: pd.DataFrame, bs: int):
    cfg = load_cfg()
    num_tot_device = cfg['system']['num_node'] * cfg['system']['num_device']

    r, s = df_gpu.loc[0], df_offload.loc[0]
    # xx_off: (memory/storage access) off-chip memory access
    # xx_on: (GPU-compute) on-chip memory & compute
    fc_off   = (r[BASE["fc_dram"]] + r[BASE["ffn_dram"]] + r[BASE["shared_moe_hbm"]]) / 1e9
    fc_on    = (r[BASE["fc_comp"]] + r[BASE["ffn_comp"]] + r[BASE["shared_moe_comp"]]) / 1e9
    attn_off = r[BASE["attn_dram"]] / 1e9
    attn_on  = r[BASE["attn_comp"]] / 1e9
    moe_off  = (r[BASE["moe_hbm"]] + s[OFFLOAD]) / 1e9
    moe_on   = r[BASE["moe_comp"]] / 1e9
    # system background energy
    background   = num_tot_device * BACKGROUND_POWER * latency / 1e3
    per_tok  = (fc_off + fc_on + attn_off + attn_on + moe_off + moe_on + background) / bs

    print(f"\n>> {label}-offloaded (nvlink5.0) case energy (J/single decode stage)")
    print(f"   FC off-chip      : {fc_off:.6f}")
    print(f"   FC on-chip       : {fc_on:.6f}")
    print(f"   Attn off-chip    : {attn_off:.6f}")
    print(f"   Attn on-chip     : {attn_on:.6f}")
    print(f"   MoE off-chip     : {moe_off:.6f}")
    print(f"   MoE on-chip      : {moe_on:.6f}")
    print(f"   System background: {background:.6f}")
    print(f"   ---------------------------")
    print(f"   Per token        : {per_tok:.6f} (÷ batch_size={bs})")


def compute_latency_baseline(label: str, df_base: pd.DataFrame) -> float:
    r = df_base.loc[0]
    # FC
    fc_sum = sum(r[col] for col in LATENCY_FC_COLS)
    fc_lat = fc_sum / 1e6
    # Attention
    attn_lat = r[LATENCY_ATTEN] / 1e6
    # MoE
    moe_lat  = r[LATENCY_MOE] / 1e6
    # Communication
    comm_lat = r[LATENCY_COMM] / 1e6
    per_tok = fc_lat + attn_lat + moe_lat + comm_lat

    print(f"\n>> {label}-latency (ms/token)")
    print(f"   FC            : {fc_lat:.6f}")
    print(f"   Attention     : {attn_lat:.6f}")
    print(f"   MoE           : {moe_lat:.6f}")
    print(f"   Communication : {comm_lat:.6f}")
    print(f"   ---------------------------")
    print(f"   Per token     : {per_tok:.6f}")

    return per_tok

def compute_latency_offload(label: str, nvlink_gen: str, df_base: pd.DataFrame, df_cpu: pd.DataFrame, bs: int) -> float:
    r, s = df_base.loc[0], df_cpu.loc[0]
    # FC
    fc_sum = sum(r[col] for col in LATENCY_FC_COLS)
    fc_lat = fc_sum / 1e6
    # Attention
    attn_lat = r[LATENCY_ATTEN] / 1e6
    # MoE
    if bs == 1024 and nvlink_gen == "nvlink5.0":
        # For this batch size and NVLink generation, compute latency exceeds gated-expert weight load latency for both Mixtral and DeepSeek-R1
        moe_lat  = r[LATENCY_MOE] / 1e6
    else:
        # For others, gated-expert weight load latency exceeds compute latency
        moe_lat  = (s[LATENCY_MOE] - (fc_sum + r[LATENCY_ATTEN] + r[LATENCY_COMM] - r[LATENCY_NON_MOE_DECODER])) / 1e6
    # Communication
    comm_lat = r[LATENCY_COMM] / 1e6
    per_tok = fc_lat + attn_lat + moe_lat + comm_lat

    print(f"\n>> {label} ({nvlink_gen}) latency (ms/token)")
    print(f"   FC            : {fc_lat:.6f}")
    print(f"   Attention     : {attn_lat:.6f}")
    print(f"   MoE           : {moe_lat:.6f}")
    print(f"   Communication : {comm_lat:.6f}")
    print(f"   ---------------------------")
    print(f"   Per token     : {per_tok:.6f}")

    return per_tok


def main():
    for model_name, bs_list in EXPERIMENTS.items():
        for bs in bs_list:
            print("\n" + "="*60)
            print(f" Experiment: model={model_name}, batch_size={bs}")
            print("="*60)

            input_sequence_length = 1024
            update_configuration(model_name, bs, input_sequence_length)

            # -----------------------------------------------------------------------------
            # Configuration flags:
            #   target_gpu (bool):
            #     • True  – measure GPU compute latency and energy
            #     • False – measure latency and energy for offloaded‐weight copying
            #   offload_cpu (bool), offload_ssd (bool):
            #     • offload_cpu=False and offload_ssd=False
            #         – baseline (no offloading)
            #     • offload_cpu=True
            #         – gated‐expert weights are offloaded to CPU memory
            #     • offload_ssd=True
            #         – gated‐expert weights are offloaded to SSD
            # -----------------------------------------------------------------------------

            f_baseline = run_sim(target_gpu=True, offload_cpu=False, offload_ssd=False)
            f_cpu_offload_gpu_compute = run_sim(target_gpu=True, offload_cpu=True,  offload_ssd=False)
            f_cpu_offload_weight_copy = run_sim(target_gpu=False, offload_cpu=True,  offload_ssd=False)
            # below, GPU and CPU-memory/SSD are connected with NVLink4.0
            f_ssd_offload_gpu_compute_nvlink4 = run_sim(target_gpu=True, offload_cpu=False, offload_ssd=True, nvlink_gen=4)
            f_ssd_offload_weight_copy_nvlink4 = run_sim(target_gpu=False, offload_cpu=False, offload_ssd=True, nvlink_gen=4)
            # below, GPU and CPU-memory/SSD are connected with NVLink5.0
            f_ssd_offload_gpu_compute_nvlink5 = run_sim(target_gpu=True, offload_cpu=False, offload_ssd=True, nvlink_gen=5)            
            f_ssd_offload_weight_copy_nvlink5 = run_sim(target_gpu=False, offload_cpu=False, offload_ssd=True, nvlink_gen=5)

            df_baseline = pd.read_csv(f_baseline)
            df_cpu_offload_gpu_compute  = pd.read_csv(f_cpu_offload_gpu_compute)
            df_cpu_offload_weight_copy  = pd.read_csv(f_cpu_offload_weight_copy)
            df_ssd_offload_gpu_compute_nvlink4  = pd.read_csv(f_ssd_offload_gpu_compute_nvlink4)
            df_ssd_offload_weight_copy_nvlink4  = pd.read_csv(f_ssd_offload_weight_copy_nvlink4)
            df_ssd_offload_gpu_compute_nvlink5  = pd.read_csv(f_ssd_offload_gpu_compute_nvlink5)
            df_ssd_offload_weight_copy_nvlink5  = pd.read_csv(f_ssd_offload_weight_copy_nvlink5)

            real_bs = load_cfg()['serving']['max_batch_size']

            latency_baseline = compute_latency_baseline("Baseline", df_baseline)
            compute_latency_offload("Offloaded-case", "nvlink4.0", df_ssd_offload_gpu_compute_nvlink4, df_ssd_offload_weight_copy_nvlink4, real_bs)
            latency_offload = compute_latency_offload("Offloaded-case", "nvlink5.0", df_ssd_offload_gpu_compute_nvlink5, df_ssd_offload_weight_copy_nvlink5, real_bs) 

            compute_energy_baseline("Baseline", latency_baseline, df_baseline, real_bs)
            compute_energy_offload("SSD", latency_offload, df_ssd_offload_gpu_compute_nvlink4, df_ssd_offload_weight_copy_nvlink4, real_bs) # SSD offload
            compute_energy_offload("CPU", latency_offload, df_cpu_offload_gpu_compute, df_cpu_offload_weight_copy, real_bs) # CPU-memory offload
    
    print("\nAll experiments done.")

if __name__ == "__main__":
    main()
