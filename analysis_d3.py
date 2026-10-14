"""D3: MoE execution flow and why expert loading is the bottleneck.

Input: run_mixtral_single.log (mixtral_single_gpu.yaml with print_log: on), which
prints the op tree of one decode step (batch 8):  name | time | ... | in -> out.
"""
import re
from collections import defaultdict

HBM_BW, PEAK_FLOPS, BYTES = 3.352e12, 989.4e12, 2      # H100, FP16
OP = re.compile(r"^(\t*)(\S+) \| ([\d.e+-]+)us .*?\| (.*?) -> (.*?) \|")
SHAPES = re.compile(r"tensor\((\d+), (\d+)\)")

ops = []                                                # (depth, name, us, in, out)
for line in open("run_mixtral_single.log"):
    m = OP.match(line)
    if m:
        ops.append((len(m[1]), m[2], float(m[3]),
                    SHAPES.findall(m[4]), SHAPES.findall(m[5])))

# 1. Flow of one MoE layer: the direct children of the first expertFFN node.
start = next(i for i, o in enumerate(ops) if o[1] == "expertFFN")
depth = ops[start][0]
print(f"[1] MoE flow, layer 0 (expertFFN = {ops[start][2]:.1f} us):")
for d, name, us, _, out in ops[start + 1:]:
    if d <= depth:
        break
    if d == depth + 1:
        extra = (" tokens/expert = " + str([int(s[0]) for s in out])) if name == "moe_route" else ""
        print(f"    {name:<28s} {us:9.3f} us{extra}")

# 2. Roofline of one expert projection (gate_proj of the first active expert).
d, name, us, inp, out = next(o for o in ops if o[1] == "gate_proj" and o[2] > 0)
m, k, n = int(inp[0][0]), int(inp[0][1]), int(out[0][1])
t_mem = k * n * BYTES / HBM_BW * 1e6
t_cmp = 2 * m * k * n / PEAK_FLOPS * 1e6
print(f"\n[2] gate_proj [{m}x{k}] x [{k}x{n}]: measured {us:.3f} us")
print(f"    weight read {k * n * BYTES / 1e6:.1f} MB / 3.352 TB/s = {t_mem:.3f} us")
print(f"    {2 * m * k * n / 1e6:.1f} MFLOP / 989.4 TFLOPS      = {t_cmp:.3f} us")
print(f"    simulator takes max() -> memory-bound, weight read is {t_mem / t_cmp:.0f}x the math "
      f"(Op/B = {m} token(s) vs H100 ridge {PEAK_FLOPS / HBM_BW:.0f})")

# 3. All 32 layers: share of time in experts and experts actually used.
moe_us = sum(o[2] for o in ops if o[1] == "expertFFN")
layer_us = sum(o[2] for o in ops if o[1].startswith("MoE_decoder_"))
active = [sum(int(s[0]) > 0 for s in o[4]) for o in ops if o[1] == "moe_route"]
print(f"\n[3] expertFFN = {100 * moe_us / layer_us:.1f}% of decoder time over {len(active)} layers")
print(f"    experts with >=1 token per layer: mean {sum(active) / len(active):.2f} of 8 "
      f"-> a decode step streams most of the expert weights every layer")
