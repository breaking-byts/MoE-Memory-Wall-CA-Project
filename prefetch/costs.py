"""Operation-level cost model of Mixtral-8x7B decode with experts in off-chip DRAM.

Every formula and constant here is copied from the SSD-offloading fork of
LLMSimulator (paths relative to SSD-offloading/LLMSimulator/):

  expert execution  src/hardware/linear_impl.cpp  LinearExecutionGPU, collects_gpu_compute: on
                    t = max(2mkn / peak_flops, (mk + kn + mn) * B / (HBM_bw - link_bw))
  expert loading    same function, collects_gpu_compute: off
                    t = kn * B / link_bw            (per projection, 3 per expert)
  activation        src/hardware/activation_impl.cpp (SiLU * up, memory bound)
  router            gate_fn is a linear inside the CSV's expert_ffn column
  copy energy       src/module/module_graph.cpp: (9.2 DDR5 read + 4.2 HBM write) pJ/bit
  background power  figure_4_and_5.py: 70 W (H100 incl. HBM) + 57 W (256 GB DDR5)

`experiments/validate_costs.py` checks this model against the simulator's own
per-step CSV rows (expert_ffn in both modes, moe_offload energy) -- they agree
to floating-point precision, so the timeline simulator below is driven by the
same numbers the simulator produces.
"""
from dataclasses import dataclass, field

GiB = 1024 ** 3


@dataclass(frozen=True)
class Platform:
    name: str
    peak_flops: float          # FP16, FLOP/s
    hbm_bw: float              # B/s
    hbm_capacity: float        # bytes
    link_bw: float             # off-chip DRAM -> HBM, B/s
    link_name: str
    # energy (framework constants)
    copy_pj_per_bit: float = 9.2 + 4.2      # DDR5 read + HBM write
    background_w: float = 70.0 + 57.0       # GPU(+HBM) + 256 GB DDR5 static/idle power
    # SSD-offloading subtracts the copy bandwidth from HBM bandwidth for GPU work
    contention: bool = True

    @property
    def exec_bw(self):
        return self.hbm_bw - self.link_bw if self.contention else self.hbm_bw


H100_NVL5 = Platform("H100", 989.4e12, 3.352e12, 80 * GiB, 900e9, "NVLink-C2C 5.0 (900 GB/s)")
H100_NVL4 = Platform("H100", 989.4e12, 3.352e12, 80 * GiB, 450e9, "NVLink 4.0 (450 GB/s)")
A100_NVL4 = Platform("A100", 312e12, 2.039e12, 80 * GiB, 450e9, "NVLink 4.0 (450 GB/s)")
# Not a simulator option (offload_bandwidth only offers 4/5); used only for the
# link-bandwidth sensitivity sweep, with the same formulas.
H100_PCIE5 = Platform("H100", 989.4e12, 3.352e12, 80 * GiB, 64e9, "PCIe 5.0 x16 (64 GB/s)")
PLATFORMS = {"H100-NVL5": H100_NVL5, "H100-NVL4": H100_NVL4,
             "A100-NVL4": A100_NVL4, "H100-PCIe5": H100_PCIE5}


@dataclass(frozen=True)
class MoEModel:
    name: str = "mixtral-8x7b"
    hidden: int = 4096
    inter: int = 14336
    layers: int = 32
    experts: int = 8
    top_k: int = 2
    bytes_per: int = 2                       # FP16
    non_expert_weight_gib: float = 2.99      # 86.99 GiB weights - 256 experts (midterm log)

    @property
    def expert_bytes(self):                  # gate + up + down projections
        return 3 * self.hidden * self.inter * self.bytes_per


MIXTRAL = MoEModel()


def _linear_exec(m, k, n, plat, model):
    flops = 2.0 * m * k * n
    mem = (m * k + k * n + m * n) * model.bytes_per
    return max(flops / plat.peak_flops, mem / plat.exec_bw)


def expert_exec_s(m, plat, model=MIXTRAL):
    """GPU time to execute one expert on m tokens (weights already in HBM):
    gate/up/down projections + the SiLU*mul activation (activation_impl.cpp:
    reads gate_out and its input, writes the output, each m x inter, memory bound)."""
    if m <= 0:
        return 0.0
    h, f = model.hidden, model.inter
    act = 3 * m * f * model.bytes_per / plat.exec_bw
    return 2 * _linear_exec(m, h, f, plat, model) + _linear_exec(m, f, h, plat, model) + act


def gate_s(batch, plat, model=MIXTRAL):
    """Router (gate_fn) GEMV, [B x hidden] x [hidden x E]; part of the CSV's expert_ffn."""
    return _linear_exec(batch, model.hidden, model.experts, plat, model)


def expert_load_s(plat, model=MIXTRAL):
    """Time to copy one expert DRAM -> HBM over the link."""
    return model.expert_bytes / plat.link_bw


def copy_energy_j(nbytes, plat):
    return nbytes * 8 * plat.copy_pj_per_bit * 1e-12


def max_cache_experts(plat, model=MIXTRAL, kv_act_gib=1.5):
    """Experts that fit in HBM after non-expert weights, KV cache and activations."""
    free = plat.hbm_capacity / GiB - model.non_expert_weight_gib - kv_act_gib
    return int(free * GiB // model.expert_bytes)
