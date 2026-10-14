# CA Project 2 (MoE Memory Wall) — Midterm Reproduction & Understanding Guide

> Purpose: let any group member reproduce every result from scratch and
> understand it well enough to defend in viva. Read top to bottom; each part
> ends with "check you understand" prompts.
>
> NOTE on course rules: this guide is for *your understanding*. The final
> report/PPT must still be written by you in your own words.

Midterm asks 4 things, and here is where each lives:
| # | Deliverable | Evidence produced |
|---|-------------|-------------------|
| D1 | Reproduce Duplex, working sim | `mixtral_gpu.yaml` vs `mixtral_duplex.yaml` + logs + 2 CSVs |
| D2 | Single-GPU + off-chip DRAM; expert load vs exec time | `mixtral_single_gpu.yaml` + `ssd_results/` (5 runs) + `analysis_d2.py` |
| D3 | MoE execution flow + why loading is the bottleneck | `run_default.log` per-op trace + `analysis_d3.py` + Sec. 6 |
| D4 | Temporal/spatial correlation + plots/stats | skew run + `analysis_d4.py` + `d4_*.png` + trace-driven run |

Read the ⚠️ boxes — each one marks a claim that was wrong or unsourced at some
point and is now either fixed or explicitly caveated. In a viva, the grader may
well probe exactly those.

---

## 1. Concepts you must own (viva foundation)

**1.1 Op/B (arithmetic intensity).** `Op/B = FLOPs done / bytes loaded from DRAM`.
For one FC weight matrix `W[M×N]` (FP16) with T tokens: you load `2·M·N`
bytes once and do `2·T·M·N` ops → `Op/B ≈ T`. Prefill has
`T = batch·Lin` (tens of thousands, GEMM — GPU happy). Decoding has
`T = batch·1` (GEMV — GPU starves). MoE is worse: those few tokens scatter
over 8–64 experts, so per-expert T ≈ 2–4 → Op/B ≈ 2–4.

**1.2 Continuous batching.** Prefill of new requests is packed with decoding
of old ones, stage by stage. Two stage types: *mixed* (contains a prefill)
and *decoding-only*. Since each request has 1 prefill + Lout decodes,
decoding-only dominates. Latency terms: **TBT** (streaming speed),
**T2FT** (prompt → first token), **E2E** (arrival → done);
**p50/p90/p99** = median/90th/99th percentile — tail matters because one slow
mixed stage stalls the whole batch (lockstep).

**1.3 Why naive hetero fails.** Putting all MoE on a weak high-bandwidth
device is great for decoding-only (low Op/B) but a mixed stage suddenly gives
each expert ~500 tokens (high Op/B) → the weak device chokes → p90/p99
explode. Duplicating MoE weights on both sides wastes HBM (MoE ≈ 70% of
params), shrinking KV space → smaller batches → lower throughput. Hence
Duplex: *both engines sharing one HBM* — no duplication.

**1.4 Logic-PIM in one paragraph.** HBM pseudo-channel: ~32 banks share one
set of I/O wires → only 1 bank readable at a time. Old Bank-PIM puts ALUs in
the DRAM die (great for Op/B<1) but DRAM transistors are ~10× bigger than
logic ones, so reaching MoE's Op/B≈8 costs 20–27%+ die area. Logic-PIM puts
ALUs on the *logic die* (dense) and adds 4× TSVs (pitch shrank 50→22µm) to
feed them: read 8 banks at once ("bank bundle") at `tCCD_L` spacing = net 4×
bandwidth, compute matched at Op/B=8 (21.3 TFLOPS/stack). xPU and Logic-PIM
hit *different* bundles concurrently — no conflict if weights/KV are striped
across bundles.

**1.5 Co-processing + parallelism.** Experts are independent → route
few-token experts to Logic-PIM, many-token ones to xPU (lookup-table
decision, negligible overhead). Prefill attention → xPU, decode attention →
Logic-PIM. *Tensor parallelism* (shard each matrix; all-reduce) keeps every
device in every expert → splittable; *expert parallelism* (whole experts per
device; all-to-all) strands 1–2 experts/device → unsplittable. Rule:
EP across nodes (slow IB), TP within a node (fast NVLink). That's why
Duplex+PE+ET beats Duplex in the paper.

**1.6 Your project's twist.** Same bottleneck as Duplex, different fix: one
GPU, all experts in slow off-chip DRAM (CPU/SSD), and *hide* the load with a
training-free predictor + prefetcher instead of extra bandwidth.

✅ Check: can you explain, without notes, why decoding is GEMV while prefill
is GEMM, and why p99 matters more than p50 for chat?

---

## 2. Environment setup (do exactly this)

You are on macOS ARM + Apple clang. The sim needs Linux + g++-11.4. Docker
Desktop is absent but `podman` exists → we run Ubuntu 22.04 containers in a
podman VM. Your files live on the Mac; containers see them at `/work`.

```bash
cd /Users/leelanshkharbanda/CA_Project
podman machine init --cpus 4 --memory 8192 --disk-size 40 ca-vm
podman machine start ca-vm
podman pull docker.io/library/ubuntu:22.04
```

Pattern for every containerised step (note `-w` workdir, `:z` volume flag):

```bash
podman run --rm -v "$PWD:/work:z" -w /work/LLMSimulator/build \
  docker.io/library/ubuntu:22.04 bash -c '<command>'
```

⚠️ Quoting trap we hit: inside `bash -c "..."`, the Mac shell expands `$var`.
Use single quotes around the inner script (`bash -c '... $m ...'`) or escape
(`\$?`). Our first SSD loop silently ran the default config 3× because of this.

✅ Check: `podman machine list` shows `ca-vm` running; `podman run --rm
ubuntu:22.04 echo hi` prints hi.

---

## 3. Clone, patch, build

```bash
git clone https://github.com/scale-snu/LLMSimulator.git
cd LLMSimulator
git submodule update --init --recursive   # pulls Ramulator2; empty dir = you skipped this
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch   # adds bank-bundle/PIM support (whitespace warnings are normal)
cd ../../../
```

Build (inside container; `apt` + configure + make in ONE run because
containers are ephemeral):

```bash
podman run --rm -v "$PWD:/work:z" -w /work/LLMSimulator docker.io/library/ubuntu:22.04 bash -c \
  "apt update -qq && apt install -y -qq g++-11 cmake clang git python3 >/dev/null 2>&1; \
   rm -rf build && mkdir build && cd build && \
   cmake -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_STANDARD=17 .. && \
   make -j4"
```

**The build bug you must understand:** first `make` failed with
`no member named 'filesystem'/'string_view'/'optional' in namespace 'std'`
(`make_full.log`). Root cause: the CMakeLists never sets a C++ standard and
CMake picked `/usr/bin/c++` (= clang) defaulting to an old standard, while the
code needs C++17. Fix = force `g++-11` (whose default is gnu++17) *and*
`-DCMAKE_CXX_STANDARD=17`. If you ever see `std::`-missing errors, think
"wrong compiler/standard" first.

Verify: `LLMSimulator/build/run` exists (~190KB). Smoke test with defaults:

```bash
podman run --rm -v "$PWD:/work:z" -w /work/LLMSimulator/build docker.io/library/ubuntu:22.04 \
  bash -c './run > /work/run_default.log 2>&1; echo EXIT=$?'
# expect EXIT=0; run_default.log shows per-op lines like:
# gate_proj | 1.8373us | ... | Op/B 3.995 ... GPU
```

✅ Check: explain why `apt install` must be in the same `podman run` as `make`.

---

## 4. D1 — reproduce Duplex (config field by field)

`eval/test.cpp` is the entry: reads the yaml, picks `ModelConfig` (mixtral /
glam / grok1… in `src/model/model_config.h`) and `SystemConfig` (A100/H100/
B100/B200 in `src/hardware/hardware_config.h`), then scheduler → cluster →
model → `runIteration`. Code map: `src/scheduler/` = continuous batching;
`src/module/expert.cpp, route.cpp, attention.cpp` = MoE/GQA layers;
`src/hardware/*_impl.cpp` = roofline timing (FLOPs/BW); `src/dram/` +
patched Ramulator2 = Logic-PIM memory. `processor_type: GPU+LOGIC` +
`parallel_execution: on` + `expert_tensor_degree: 4` = Duplex+PE+ET;
`GPU` + `off` + `1` = baseline. `TP/DP` in output filenames derive from
`num_device / tensor_degree`. `nvlink_gen` must be 4 (H100) or 5 (B100/B200)
or the run aborts; `decode_mode`/`prefill_mode` are mutually exclusive.

| file | model | system | serving/sim |
|---|---|---|---|
| `mixtral_gpu.yaml` | mixtral | H100, 1×4, GPU, TP1, PE off | batch 32, 1024/128, FP16, iter 3 |
| `mixtral_duplex.yaml` | mixtral | H100, 1×4, GPU+LOGIC, TP4, PE on | same |

```bash
podman run --rm -v "$PWD:/work:z" -w /work/LLMSimulator/build docker.io/library/ubuntu:22.04 \
  bash -c './run /work/mixtral_gpu.yaml > /work/run_mixtral_gpu.log 2>&1; ./run /work/mixtral_duplex.yaml > /work/run_mixtral_duplex.log 2>&1'
mkdir -p LLMSimulator/data   # the sim does NOT create it; without it the CSV is silently skipped!
```

### Results — and where each number actually lives

Each log is only ~8 lines: the ACT/Weight/Cache breakdown, one footprint
`Total:`, one wall-clock `Total:`, and the path of the CSV it wrote. **Every
component number is in that CSV, not the log.** When you quote a number,
say which file it came from — that is the first thing a grader will check.

| | GPU baseline | Duplex+PE+ET | source |
|---|---|---|---|
| Total (wall clock) | 23.94 ms | 7.73 ms → **3.10×** | log `Total:` |
| per-iter latency | 7.98 ms | 2.57 ms → **3.10×** | CSV `latency` |
| expert_ffn | 6765 µs | 1721 µs (3.9×) | CSV `expert_ffn` |
| atten / o_proj / qkvgen | 337 / 322 / 482 µs | 81 / 83 / 124 µs | CSV `atten_gen` / `o_proj` / `qkvgen` |
| footprint | 26.07 GiB | 23.96 GiB | log `Total:` |
| total energy | 3.228 **J** | 3.160 **J** (−2.1%) | CSV `total_energy` |

Units: the sim divides by `1024³`, so footprints printed as "GB" are really
**GiB**. Energy is in **J**, not "Ge" — `total_energy` is in nJ
(`common/type.h:10`: `energy_nJ`), so `3228401050` nJ = 3.2284 J. Say J.

CSV columns to know: `latency`, `expert_ffn`, `atten_gen`, `qkvgen`,
`moe_dram/moe_comp` (energy split), `OOM`. Slightly above the paper's ~2.5×
because this slice is decode-heavy — say that, don't hide it.

Two honest caveats to state in the report:
- **Energy is not a win.** 3.228 J → 3.160 J is 2.1%; Duplex's benefit here is
  latency, not energy. Don't present the delta as a saving.
- **`iter: 3` yields 2 CSV rows.** `cluster.cpp:468-474` calls `exportToCSV`
  *before* `run()` on each pass, so the last iteration's data is never
  written. (CSV header is at `cluster.cpp:428-435`.) The two rows agree
  (7.980/7.981 ms GPU; 2.574/2.584 ms duplex), so nothing is cherry-picked —
  but don't claim 3 iterations of data.
- **The pair changes three fields at once** (`processor_type`, both tensor
  degrees, `parallel_execution`). That's Duplex+PE+ET as a bundle, matching the
  paper's configuration, but you cannot attribute the 3.10× to PE alone from
  this pair.

✅ Check: point to where `parallel_execution` flows from yaml into the device
(`eval/test.cpp:99` → `hardware_config.h` → `executor.cpp:153`).

---

## 5. D2 — single GPU + off-chip (two evidences)

**5a. Main sim.** `mixtral_single_gpu.yaml`: 1×1, GPU-only, TP1, batch 8,
512/32. Result: footprint **88.48 GiB > 80 GiB → OOM=1** (your motivation
sentence). Gotcha: `gpu_gen: A100` crashes (`YAML::BadFile`) because
`src/hardware/device.cpp:33-40` only wires DRAM configs for H100/B100/B200 —
so H100 stands in as the "A100-class 80 GiB" reference (`hardware_config.h:213`
is `80.0 * 1024³`). State this in the report.

⚠️ The `A100` crash is **code-verified only** — no log or yaml in this
workspace captures the `YAML::BadFile` message. `build/` ships only
`dram_config_HBM3_80GB.yaml` and `dram_config_HBM3E_192GB.yaml`, and
`device.cpp:33-40` builds an empty path for any other `gpu_gen`. To cite it
defensibly, add `gpu_gen: A100` to a copy of `mixtral_single_gpu.yaml`, run it,
and keep the error output as an artifact.

⚠️ **Do not use this run's timing as an execution baseline.** The yaml sets
`exit_out_of_memory: off`, so `cluster.cpp:241-244` sets the OOM flag and the
run *continues*, holding 88.48 GiB on an 80 GiB device with bandwidth
unchanged. Its `expert_ffn = 23.16 ms` is reported but physically impossible.
Use it for the footprint/OOM argument; use §5b for timings. `analysis_d2.py`
prints this warning alongside the numbers.

Expert-size math (Mixtral: hidden 4096, inter 14336, 3 projections, FP16 —
`model_config.h:91`): per expert = 3·4096·14336·2B = **0.352 GB** (0.328 GiB);
top-2/layer = 0.705 GB; all 8/layer = 2.82 GB; 32 layers top-2 = 22.55 GB
(= exactly 21.000 GiB — that exactness is the giveaway that these are the
sim's binary units, while the sim's *printed* footprints use 1024³).

Load time to bring one layer's top-2 experts into HBM (regenerate with
`./.venv/bin/python analysis_d2.py`):

| tier | bandwidth | load top-2/layer | vs HBM |
|---|---|---|---|
| HBM3 (on-package) | 3.352 TB/s | 0.21 ms | 1.0× |
| **C2C 5.0 — what the sim models** | 900 GB/s | 0.78 ms | 3.7× |
| C2C 4.0 — also modellable | 450 GB/s | 1.57 ms | 7.4× |
| CPU DDR5 *(literature, not modelled)* | 32 GB/s | 22.02 ms | 105× |
| M.2 NVMe SSD *(literature, not modelled)* | 7 GB/s | 100.66 ms | 479× |

⚠️ The last two rows are **not simulator settings**. SSD-offloading has no
CPU-DDR or SSD bandwidth path — only the C2C link. Use those rows solely to
explain that a real PCIe SSD would look worse than what we simulated. Never
label them "measured".

**5b. SSD-offloading framework** (the spec's designated tool for exactly this).
`SSD-offloading/patches.sh` applies `ramulator2_pim.patch` +
`ssd_offloading.patch` (7364 lines: new eval apps, `figure_*.py`, offload
model) onto its pinned LLMSimulator; build identically. Learn its model —
it's viva gold. All paths below are relative to
`SSD-offloading/LLMSimulator/src/` (the patched copy, whose line numbers differ
from the unpatched `LLMSimulator/src/`): `hardware/linear_impl.cpp:28-77` (the
latency fork), `module/module_graph.cpp:258-282` (energy constants),
`hardware/hardware_config.h:97` (bandwidth subtraction), `:170-186` (offload
link bandwidth), `dram/power.h:32-36` (base HBM/MAC energy constants).

Reproduction trap: `linear_impl.cpp:47`, `hardware_config.h:170` and
`module_graph.cpp:267` all read `"config.yaml"` from the **current working
directory**, ignoring `argv[1]`. So `./run ssd_cfgs/ssd.yaml` silently uses
`build/config.yaml` for the offload flags. Copy the intended config over
`build/config.yaml` first (that is why `ssd_results/*.yaml` are byte-identical
copies of `ssd_cfgs/*.yaml`).
- `collects_gpu_compute: on` (default): MoE latency uses GPU roofline but HBM
  BW is *reduced* by the C2C copy BW (`base − offload_bandwidth`) — models
  contention of concurrent copies. `moe_offload` is structurally **0** here
  (`linear_impl.cpp:95-99` never sets `moe_offload_read_count`), so those rows
  are for **latency only**. Don't read energy from them.
- `collects_gpu_compute: off`: `expert_ffn` = pure copy time
  (`k*n*precision / offload_bandwidth`) and `compute_duration` is **never
  set** — compute is discarded by construction in exactly the mode that reports
  load time (`linear_impl.cpp:60-62`).
- Both CPU/SSD flags share one latency formula (C2C 4.0 = 450 or 5.0 = 900
  GB/s via `offload_bandwidth: 5`); tiers differ **only in energy**.

⚠️ **The spec asks for load time *and* execution time. No single run reports
both.** `expert_ffn` is copy time in `*_copy` mode and fused roofline exec in
the other three. Presenting 6.52 ms and 24.28 ms in one column implies they are
one comparable metric; they come from two mutually exclusive code paths over
different byte counts. Say which mode produced which number.

Runs (`ssd_cfgs/*.yaml`, batch 4, decode → `ssd_results/*.csv`). Regenerate
this table and re-check it against the constants with
`./.venv/bin/python analysis_d2.py --check`:

| mode | expert_ffn (32 layers) | movement energy | compute |
|---|---|---|---|
| HBM exec (`hbm.yaml`) | 6.52 ms | 2.06 J (`moe_hbm`) | 0.248 J |
| CPU copy-load (`cpu_copy.yaml`) | 24.28 ms (**3.72×**) | 6.57 J (3.19×) | 0.248 J |
| SSD copy-load (`ssd_copy.yaml`) | 24.28 ms (**3.72×**) | 52.29 J (**25.38×**) | 0.248 J |
| concurrent-copy contention (`cpu.yaml`/`ssd.yaml`) | 6.52 → 8.92 ms (**+36.7%**) | not counted | 0.248 J |

**The 25× derivation** (`module_graph.cpp:258-282`) — get this right, it was
previously written as `106.6/13.4`, which is only 7.96×:

| path | pJ/bit | |
|---|---|---|
| HBM3 read | 4.2 | ← the baseline any movement ratio is measured against |
| DDR5 read | 9.2 | raw medium only |
| CPU→HBM | 9.2 + 4.2 = **13.4** | + HBM write-back |
| SSD read | 102.4 | raw medium only |
| SSD→HBM | 102.4 + 4.2 = **106.6** | + HBM write-back |

- SSD vs HBM = 106.6/4.2 = **25.4×** ✓ (measured 25.37×)
- CPU vs HBM = 13.4/4.2 = **3.19×** ✓ (measured 3.19×)
- SSD vs CPU = 106.6/13.4 = 7.96×, and raw-media 102.4/9.2 = 11.1×. Neither
  is the 25×. The expert lands in HBM before compute starts, so the HBM write
  belongs in the numerator and HBM read is the like-for-like baseline.

**Why CPU and SSD latencies are identical but energies differ:** both tiers copy
over the same C2C link (900 GB/s), so latency depends only on bytes/link
bandwidth. Energy depends on *where* the bytes come from: 9.2 vs 102.4 pJ/bit.
`analysis_d2.py --check` asserts all of this, including that the three tiers
move the same 61.3 GB and that contention equals 3.352/2.452 = 1.367×.

Note honestly: copy link is C2C 900 GB/s, so a real PCIe SSD would look worse
— your numbers are conservative.

---

## 6. D3 — MoE flow, and *why loading* is the bottleneck

**6a. The flow.** For one MoE layer (`expert.cpp:140-190`):

1. **Gate** — `gate_fn`, a `ColumnParallelLinear(hidden → num_routed_expert)`
   scores every expert (`expert.cpp:29-32`). Mixtral: 8 experts.
2. **Dispatch** — `moe_scatter` (`expert.cpp:152`) permutes tokens by expert,
   then `moe_route` = `Route::forward` (`route.cpp:215`) builds the per-expert
   input lists. ⚠️ `route.cpp:aggregate_expert` (`route.cpp:149`) is *not* this
   step — it is per-batch bookkeeping that tallies `num_token_in_expert[]` and
   costs 0.0000 µs in the trace. Don't cite it as the dispatch.
3. **Expert FFNs** — per expert, `FeedForward3Way::forward`
   (`layer.cpp:415-437`): `gate_proj` → `act_fn` → `up_proj` → `down_proj`.
4. **Combine** — `moe_all_reduce_for_e_tp`, then `moe_gather`
   (`expert.cpp:103-110, 183-190`). ⚠️ `moe_gather` is the **combine** stage,
   not dispatch. The real symbol names are `moe_all_reduce_for_e_tp` and
   `moe_all_reduce_for_gather`; there is no bare `moe_all_reduce`. `sync_*`
   barriers bracket all four stages and are 0.0000 µs except `sync_2`.

**6b. Why "expert_ffn is 96% of runtime" does not prove a load bottleneck.**
`linear_impl.cpp:61-62` reports one fused number:
`total_duration = max(compute_duration, memory_duration)`. There is no
`expert_load` or `expert_exec` column. So a 96% share is equally consistent
with being 96% weight streaming, 96% arithmetic, or a mix. Worse, the "0.72 ms
of math" figure in older drafts was `expert_ffn / 32 layers` — the already
*fused* number — which makes the argument circular: the load must precede a
quantity that already contains the load.

**6c. The separation, done properly.** `analysis_d3.py` produces it two ways.

*Analytically, per op* — replay the roofline for the widest `gate_proj` in
`run_default.log` (`[1, 7168] → [1, 2048]`, 1 B, B100):

| term | calculation | predicted | measured |
|---|---|---|---|
| weight read | 7168 × 2048 × 1 B = 14.68 MB ÷ 8.0 TB/s | 1.8350 µs | **1.8362 µs** |
| arithmetic | 2 × 1 × 7168 × 2048 = 29.4 MFLOP ÷ 1750 TFLOPS | 0.0168 µs | — |
| `max(...)` | `linear_impl.cpp:61-62` | 1.8350 µs | ✓ |

→ **load : compute = 109 : 1** for a single projection, Op/B = 2.00 (the trace's
own Op/B column confirms it).

*From energy, independent of the latency model* — summed over all 7296 expert
projections in `run_default.log`: DRAM-side (RD+WR+ACT) = **97.5%**, MAC =
2.5%. Moving the weights costs **40×** the arithmetic. The CSV agrees at
coarser granularity: `moe_dram / moe_comp = 53.7×` on the Mixtral single-GPU run.

At the tier level (`ssd_results/`, §5b): **load is 3.72× execution** and HBM
movement is **8.3×** compute energy even when everything is already on-chip.

**6d. Model identity — do not mix these up.** `run_default.log` is the
simulator's *default* config (`LLMSimulator/config.yaml`): **deepseekV3**,
hidden 7168, 60 layers, 256 experts (32/device), top-8, `precision_byte: 1`,
B100 8 TB/s, first 3 layers dense. It is **not Mixtral** (hidden 4096, 32
layers, 8 experts, top-2, FP16, H100). Never quote its ~1.84 µs as a Mixtral
timing; `analysis_d3.py` detects and labels the model for you.

Also in that trace: only **7 distinct latency values** exist across 1824
projections, 38% of expert projections are skipped (that expert got 0 tokens),
so the mean is 1.147 µs while the median for an *active* expert is 1.836 µs.
And coverage is **32/32 experts per layer** — see §7d for why that matters.

✅ Check: given only `expert_ffn` and `moe_dram`, could you tell load-bound
from compute-bound? (No.) Now show me the per-op split that can.

---

## 7. D4 — correlation (the subtlest part)

**7a. Skew (spatial), measured in-sim.** `src/scheduler/sequence.cpp:168-192`:
weights `1/(k+1)^s` at `sequence.cpp:168-171`, Zipfian draw when
`skewness>0` at `sequence.cpp:191-196`, else uniform. The draws themselves are
in `scheduler.cpp:165-177` (`getZipfianRandomExpert`); `141-163` is
`getRandomExpert` + `getEquallyDistributedExpert`, which is *not* the Zipfian
path. `mixtral_single_skew.yaml` (s=1.5): 61.16 vs 75.25 ms uniform
(**1.23×**) — hot experts share weights, less traffic.

**7b. Trace format — and the bug that hides here.**
Reverse-engineered from `scheduler.cpp:511-608` (`initExpertList`; the file is
609 lines, so an earlier "511-660" citation pointed past the end). File
`expert_data/experts_<model>_<dataset>.csv`, selected via `simulation.data:
<dataset>`. Grammar: `seq_N` opens a block → `Lin <n>` → `Sum` → prefill pairs
→ `Gen` → decode pairs → `Lout <n>`.

⚠️ **`Lin` must precede `Sum`** — `scheduler.cpp:570` does
`resize(Lin * num_layer)` on a fresh `SequenceInfo` whose `Lin` is
uninitialized (`sequence.h:21`), so reordering gives
`vector::_M_default_append`. **`Lout` must follow `Gen`** — `:567` is what
pushes the completed sequence.

⚠️ **The two sections are read in DIFFERENT orders.** This is the single
easiest thing to get wrong, and getting it wrong is silent:

| section | C++ | file rows must be |
|---|---|---|
| `Sum` (prefill) | `scheduler.cpp:577-583`: `token_id = idx % Lin`, `layer_id = idx / Lin`, store at `token_id*num_layer + layer_id` | **layer-major** — layer 0 for all Lin tokens, then layer 1… |
| `Gen` (decode) | `scheduler.cpp:585`: `push_back` in file order | **token-major** — all layers of token 0, then token 1… |

Write `Sum` token-major (the intuitive reading, and what the old draft called
"token-major") and the file parses, the run exits 0, and the simulator reads a
**transposed** trace: prefill layer-to-layer overlap collapses from 0.686 to
0.252 — exactly the random null. Meanwhile any offline analysis of the same
file still looks strongly correlated, because the Python loader shared the
writer's convention. **The old round-trip check could not catch this**: it
compared `gen_trace.py` against `analysis_d4.load_trace_csv`, both of which
used the same wrong order.

`test_trace_io.py` now ports the C++ reader and asserts the on-disk file
round-trips to the generator's in-memory array under it, plus that the Python
loader agrees with that port. Run it before trusting any D4 number:
`./.venv/bin/python test_trace_io.py`.

**7c. Correlated trace + honest methodology.** `gen_trace.py` bakes in
ST-MoE-like stats (Zipf base s=1.0, layer-stay 0.6, token persist 0.5 prefill /
0.7 decode) → `LLMSimulator/expert_data/experts_mixtral_stmoe.csv`
(64 seqs × (64 prefill + 16 decode) tokens × 32 layers × top-2). It asserts its
own round-trip through the C++ port before printing anything.

`analysis_d4.py` computes frequency bars, layer×expert heatmap,
`P(e_{l+1}|e_l)`, predictor overlap, coverage, and phase-split persistence:

| | overlap | random null |
|---|---|---|
| uniform | 0.251 | 0.25 |
| **stmoe trace (as the sim reads it)** | **0.674** | 0.25 |

The null is `TOPK/N_EXPERT = 2/8 = 0.25`, exact for real routing (which always
picks 2 *distinct* experts, `std::set` at `sequence.cpp:190-197`). `gen_uniform`
samples without replacement to match; an earlier version sampled *with*
replacement, whose true null is 0.2195 — the uniform result then sat exactly on
its own wrong baseline.

Temporal-across-tokens is now reported **per phase** rather than as one scalar,
because the phase split is the interesting result:

| phase | persistence |
|---|---|
| prefill | 0.533 |
| decode | **0.713** |

Decode persistence is higher — exactly the regime a prefetcher targets — and
measuring it requires a phase mask, since persistence across a prefill/decode
or `seq_N` boundary is meaningless. `stats()` also no longer slices to a
module-level `N_TOK`: it previously measured only the first 2000 of 5120
tokens, which were all prefill, so the decode figure was never computed.

**7d. Trace-driven sim + the key insight.** `mixtral_stmoe.yaml` (`data:
stmoe`) runs EXIT=0. `expert_ffn` 24.311 ms (uniform, matched 64/16) vs
24.206 ms (stmoe trace) — **no win**; the two configs differ by <0.5%.

⚠️ Those two numbers are the `expert_ffn` column, not end-to-end `latency`
(25.142 vs 25.037 ms — same conclusion). And the trace CSV is named
`...512_32...` because the filename comes from the yaml while Lin/Lout come
from the trace. State it rather than implying matched end-to-end runs.

Mechanism, now computed by `analysis_d4.coverage()` rather than asserted:
distinct experts touched per layer at batch 8 = **7.09 (stmoe trace) vs 7.20
(uniform)** out of 8 — the opposite assignment from earlier drafts, but the
conclusion is unchanged: **coverage saturates**. At batch 8 a decode step
already touches ~7 of 8 experts, so there is no traffic left for correlation
to remove. Its value is purely *predictive* (overlap 0.674) → that is the
final-phase prefetch argument, not a traffic-reduction one.

**Companion claim, now measured instead of asserted — "skew ≠ predictability"
is only half true.** `analysis_d4.ablation()` separates the two effects:

| scenario | overlap | Δ vs null |
|---|---|---|
| uniform (no skew, no correlation) | 0.251 | +0.001 |
| **skew only** (Zipf s=1.0, i.i.d. — no Markov chain) | **0.365** | **+0.115** |
| correlation only (Markov, uniform base) | 0.671 | +0.421 |
| skew + correlation (ST-MoE-like) | 0.699 | +0.449 |

Skew alone buys **+0.115** of overlap — about a quarter of the total signal —
because hot experts are shared across adjacent layers by chance. So the honest
statement is: *skew is worth something (it concentrates reuse), but correlation
is the dominant term (4× larger).* The claim that skew "doesn't say which expert
comes next" only holds in the sense that it is a far weaker predictor, not a
zero-information one.

✅ Check: explain why the skew run got faster but the correlated-trace run
didn't (coverage saturates at ~7/8 in both), and quote the ablation to say how
much of the predictability is skew vs correlation.

**7d. Trace-driven sim + the key insight.** `mixtral_stmoe.yaml` (`data:
stmoe`) runs EXIT=0. Matched-size result (64/16, batch 8): uniform 24.31ms vs
trace 24.21ms — *no win*. Mechanism, measured: distinct experts/layer at
batch 8 = 7.09 (uniform) vs 7.19 (trace) of 8 — **coverage saturates**, so
correlation can't cut baseline traffic. Its value is purely *predictive*
(overlap 0.67) → that is your final-phase prefetch argument. Companion line:
**skew ≠ predictability** — Zipf skew helps reuse but doesn't say *which*
expert comes next; the transition matrix does.

✅ Check: explain why the skew run got faster but the correlated-trace run
didn't, using the coverage numbers.

---

## 8. File inventory & demo recipe

Configs: `mixtral_gpu/duplex/single_gpu/single_skew/synth64/stmoe.yaml`,
`ssd_cfgs/` (hbm/cpu/ssd + `*_copy`). Logs: `run_default/mixtral_*.log`,
`ssd_*.log`, `make*.log`, `cmake2.log`. Data: `LLMSimulator/data/*.csv`,
`ssd_results/*.csv`. Analysis: `analysis_d2.py`, `analysis_d3.py`,
`analysis_d4.py`, `gen_trace.py`, `test_trace_io.py`; plots
`d4_uniform/skew_only/correlated/stmoe_trace/stmoe_file.png`, `.venv/`
(`pip install matplotlib numpy pandas` inside it). Repos: `LLMSimulator/`
(patched+built), `SSD-offloading/` (patched+built), `LLMSimulator/expert_data/
experts_mixtral_stmoe.csv`. Reference PDFs at top level.

Verify everything before the demo — all three are read-only and take seconds:

```bash
./.venv/bin/python test_trace_io.py                    # trace I/O vs C++ parser
./.venv/bin/python analysis_d2.py --check              # D2 constants vs CSVs
./.venv/bin/python analysis_d3.py                      # D3 per-op load/compute split
./.venv/bin/python analysis_d4.py                      # D4 plots + ablation
```

Demo (5 min): duplex pair → the two `Total:` lines (3.1×); single-GPU log →
footprint > 80 GiB + OOM=1 (and *why* its timing isn't a valid baseline);
`analysis_d2.py` → 3.72× load/exec and the 25.38× SSD energy derivation;
`analysis_d3.py` → 109:1 per-op load:compute; `d4_correlated.png` transition
diagonal vs flat uniform, then the ablation table for skew vs correlation.

⚠️ `run_default.log` is the **deepseekV3** default config, not Mixtral — say so
if you show it, and get Mixtral numbers from the CSVs.

## 9. Bridge to final phase
Predictor candidates: ST-MoE rule (next-layer = f(current layer, last token),
your measured **0.674** overlap is the baseline to beat) vs APEX-style adaptive
prefetch degree. Prefetch win = `overlap × load_latency` hidden behind
`compute`; cost = wasted bandwidth on mispredicts — priced with the same
constants as §5b (HBM read 4.2, CPU 13.4, SSD 106.6 pJ/bit).

Use the numbers that now exist: load latency 3.72× execute (§5b) or 109:1
per-op (§6c), decode-phase persistence 0.713 vs prefill 0.533 (§7c), and
coverage saturation at ~7.2/8 experts per layer at batch 8 (§7d) — which caps
what prefetching can save at batch 8, so measure the hit-rate-vs-prefetch-degree
curve at smaller batches too.

⚠️ The 0.674 is a **1-step, stationary** predictor: it uses only the current
layer. The ST-MoE rule also conditions on the last token, which is untested
here — so 0.674 is an optimistic baseline, not a ceiling. And since overlap
alone buys +0.115 (§7d ablation), a predictor must beat **0.365**, not 0.25,
to be worth more than Zipf skew.

Suggested first final experiment: implement the predictor *offline* on the
trace (hit rate vs prefetch degree) before touching the sim.
