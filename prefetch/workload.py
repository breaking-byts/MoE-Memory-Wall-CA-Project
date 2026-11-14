"""Routing workloads: load .npz traces, write the simulator's routing_trace file,
and read back the simulator's routing_log.

Workload .npz layout (all produced by workloads/*):
  ids          int8  [S, T, L, K]   routed experts (rank order) per sequence/token/layer
  profile_ids  int8  [S', T, L, K]  disjoint profiling split (only used to build count tables)
  spec{d}      uint8 [S, T, L, E]   optional, d in {0,1,2}: router_l softmax evaluated on the
                                    hidden state available d layers earlier (x255). spec0 = the
                                    layer's own router applied to its *pre-attention* input.
  prompt_len   int                  tokens 0..prompt_len-1 are prefill history; decode = the rest
  calib{d}     float [E, 20]        optional, P(routed | speculative rank, softmax bin) from the profiling split
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOADS = {
    "tinymoe-prose": "workloads/traces/tinymoe_prose.npz",
    "tinymoe-code": "workloads/traces/tinymoe_code.npz",
    "mixtral-calibrated": "workloads/traces/mixtral_calibrated.npz",
    "uniform": "workloads/traces/uniform.npz",
}


class Workload:
    def __init__(self, name):
        self.name = name
        z = np.load(os.path.join(ROOT, WORKLOADS[name]))
        self.ids = z["ids"].astype(np.int64)
        self.profile_ids = z["profile_ids"].astype(np.int64)
        self.prompt_len = int(z["prompt_len"])
        self.spec = {}
        self.calib = {}
        for d in (0, 1, 2):
            if f"spec{d}" in z.files:
                self.spec[d] = z[f"spec{d}"].astype(np.float32) / 255.0
            if f"calib{d}" in z.files:
                self.calib[d] = z[f"calib{d}"]
        S, T, L, K = self.ids.shape
        self.S, self.T, self.L, self.K = S, T, L, K
        self.E = 8

    @property
    def decode_steps(self):
        return self.T - self.prompt_len

    def batch(self, slots, steps=None):
        """Routing of a decode batch: sequences `slots` decode in lock-step.
        Returns full per-token history H [B, T, L, K] (prompt + decode) and the
        decode-only routing R [steps, L, B, K] that the simulator executes."""
        steps = steps or self.decode_steps
        H = self.ids[slots]
        R = H[:, self.prompt_len:self.prompt_len + steps].transpose(1, 2, 0, 3)
        return H, R


def write_sim_trace(R, path):
    """R [steps, L, B, K] -> simulator routing_trace (slot-major, step, layer, k)."""
    steps, L, B, K = R.shape
    with open(path, "w") as f:
        f.write(f"{B} {steps} {L} {K}\n")
        arr = R.transpose(2, 0, 1, 3).reshape(-1)
        for i in range(0, len(arr), 64):
            f.write(" ".join(map(str, arr[i:i + 64])) + "\n")


def read_sim_log(path, L, K):
    """Simulator routing_log -> R [steps, L, B, K] (in the order the simulator routed)."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    assert (rows[:, 1] >= 0).all(), "routing log contains prefill steps"
    steps = rows[:, 1].max() + 1
    B = rows[:, 3].max() + 1
    R = -np.ones((steps, L, B, K), dtype=np.int64)
    for call, step, layer, slot, *es in rows:
        R[step, layer, slot] = es
    return R
