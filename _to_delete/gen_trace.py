"""Generate an ST-MoE-like correlated expert trace in LLMSimulator's
expert_data CSV format.

Statistics baked in: Zipf-skewed bases (hot experts) + layer-to-layer
stay probability + consecutive-token persistence.

The Sum (prefill) and Gen (decode) sections are written in DIFFERENT orders
because the simulator's C++ reader consumes them differently -- see the
LLMSimulator trace-format note in analysis_d4.py, and scheduler.cpp:560-608.
This script asserts the on-disk file matches its in-memory array under that
same parser before printing any statistic.
"""
import numpy as np
from analysis_d4 import plot, load_trace_csv

OUT = "LLMSimulator/expert_data/experts_mixtral_stmoe.csv"

N_EXPERT, TOPK, N_LAYER = 8, 2, 32
N_SEQ, LIN, LOUT = 64, 64, 16
ZIPF_S, STAY, PERSIST_PRE, PERSIST_DEC = 1.0, 0.6, 0.5, 0.7
rng = np.random.default_rng(7)

w = np.array([1.0 / (k + 1) ** ZIPF_S for k in range(N_EXPERT)])
w /= w.sum()

def draw_base():
    return tuple(sorted(rng.choice(N_EXPERT, size=TOPK, replace=False, p=w)))

def step(cur):
    nxt = [c if rng.random() < STAY else rng.choice(N_EXPERT) for c in cur]
    if nxt[0] == nxt[1]:
        nxt[1] = (nxt[1] + rng.integers(1, N_EXPERT)) % N_EXPERT
    return tuple(sorted(nxt))

def gen_seq():
    toks, prev = [], None
    for t in range(LIN + LOUT):
        p = PERSIST_DEC if t >= LIN else PERSIST_PRE
        cur = tuple(prev) if (prev is not None and rng.random() < p) else draw_base()
        prev = cur
        layers, c = [cur], cur
        for _ in range(1, N_LAYER):
            c = step(c)
            layers.append(c)
        toks.append(layers)
    return toks  # [token][layer] -> (e1, e2)

seqs = [gen_seq() for _ in range(N_SEQ)]
with open(OUT, "w") as f:
    for s, toks in enumerate(seqs):
        f.write(f"seq_{s}\nLin {LIN}\nSum\n")
        # Sum section must be LAYER-major: scheduler.cpp:577-579 reads
        #   token_id = idx % Lin;  layer_id = idx / Lin
        # so row `idx` is token idx%Lin at layer idx//Lin. Writing token-major
        # here parses fine and exits 0, but the simulator then sees a transposed
        # trace whose layer-to-layer correlation is at the random null.
        for l in range(N_LAYER):
            for t in range(LIN):
                f.write(f"{toks[t][l][0]} {toks[t][l][1]}\n")
        f.write("Gen\n")
        for t in range(LIN, LIN + LOUT):
            for l in range(N_LAYER):
                f.write(f"{toks[t][l][0]} {toks[t][l][1]}\n")
        f.write(f"Lout {LOUT}\n")

arr = np.zeros((N_SEQ * (LIN + LOUT), N_LAYER, TOPK), dtype=int)
for s, toks in enumerate(seqs):
    for t in range(LIN + LOUT):
        arr[s * (LIN + LOUT) + t] = toks[t]

# Verify the file we just wrote is what the simulator's C++ parser will read,
# before trusting any statistic computed from it.
written = load_trace_csv(OUT, return_phases=True)
assert written[0].shape == arr.shape, f"{written[0].shape} != {arr.shape}"
bad = int((written[0] != arr).any(axis=2).sum())
assert bad == 0, (
    f"{bad} token-layers on disk differ from the in-memory trace. The Sum "
    "section order does not match scheduler.cpp:577-579."
)
mask = np.concatenate([np.r_[np.zeros(LIN, bool), np.ones(LOUT, bool)]
                       for _ in range(N_SEQ)])
res = plot(arr, "stmoe_trace", batch=8, is_decode=mask)
print("wrote LLMSimulator/expert_data/experts_mixtral_stmoe.csv",
      f"({N_SEQ} seqs x ({LIN}+{LOUT}) toks x {N_LAYER} layers x {TOPK})")
print("round-trip vs C++ parser: OK (0 mismatches)")
print(f"coverage @ batch 8 = {res['coverage']['mean_distinct']:.2f}/{N_EXPERT} "
      "experts/layer -> baseline expert traffic already saturated")
