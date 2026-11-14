# The MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

Course project (CA Project 2, Group 24). Mixtral-8x7B decode on a single GPU whose
expert weights live in off-chip DRAM, simulated with **LLMSimulator** (Duplex) and the
**SSD-offloading** fork. This repository contains both submissions:

* **Midterm (D1-D4)**: Duplex reproduction, the single-GPU offloaded system, the MoE
  execution-flow analysis and a first correlation study. Report:
  [`report/CA Midterm Report Group 24.pdf`](<report/CA Midterm Report Group 24.pdf>).
* **Final (F1-F4)**: training-free expert predictors, a prefetcher that copies predicted
  experts from DRAM into HBM ahead of use, and a latency / stall / traffic / energy
  comparison with on-demand execution. All final numbers: [`RESULTS.md`](RESULTS.md)
  (generated) and `experiments/results/*.csv`; figures in `experiments/figures/`.

Everything is simulation; no GPU is needed.

---

## 1. Headline results

Single H100-class GPU (80 GB HBM3, 3.35 TB/s), Mixtral-8x7B FP16, all 256 experts (352 MB each) in CPU DRAM
behind a 900 GB/s link, batch-1 decode. Latency per generated token, mean over 16 sequences x 191 decode steps
(speed-up vs on-demand + LRU in brackets; every group of the 16 is within +-0.03x of the mean - see `RESULTS.md`).
**C = 224** experts is everything that fits in 80 GB next to the non-expert weights, KV cache and activations
(73.5 GiB); **C = 64** (22.5 GB) is the same GPU when most of HBM is taken by KV cache (e.g. batch 64 x 4k-token
contexts need ~32 GB of KV).

| policy | tinymoe-prose C=224 | mixtral-calibrated C=224 | tinymoe-prose C=64 | mixtral-calibrated C=64 |
|---|---|---|---|---|
| simulator's own serial on-demand (load all routed experts, then execute; no overlap, no reuse) | 35.39 ms | 35.39 ms | 35.39 ms | 35.39 ms |
| on-demand, every routed expert re-copied (copies pipelined with execution) | 30.79 ms | 30.79 ms | 30.79 ms | 30.79 ms |
| on-demand + LRU expert cache (**baseline**) | 11.14 ms | 12.31 ms | 21.18 ms | 23.10 ms |
| ST-MoE predictor, re-implemented | 11.24 (0.99x) | 12.12 (1.02x) | 21.63 (0.98x) | 25.09 (0.92x) |
| **ours**: calibrated speculative gate + guided replacement + preemption + feedback throttle | **10.55 (1.06x)** | **11.44 (1.08x)** | **16.65 (1.27x)** | **22.19 (1.04x)** |
| oracle prefetch (perfect prediction) | 10.49 (1.06x) | 10.54 (1.17x) | 15.97 (1.33x) | 17.50 (1.32x) |
| exposed expert-load stall: baseline -> ours | 0.81 -> 0.22 ms | 1.98 -> 1.10 ms | 10.85 -> 6.32 ms | 12.77 -> 11.86 ms |
| energy per token: baseline -> ours | 2.48 -> 2.43 J (-2.1%) | 2.81 -> 2.77 J (-1.5%) | 5.06 -> 4.56 J (-10.0%) | 5.53 -> 5.61 J (+1.4%) |
| DRAM->HBM traffic per token: baseline -> ours | 1.1 -> 1.3 GB (+19%) | 2.7 -> 3.4 GB (+24%) | 13.2 -> 13.9 GB (+5%) | 15.4 -> 17.2 GB (+12%) |

* **The memory wall is first a capacity problem.** Keeping experts resident in an LRU HBM cache is worth
  1.3-3.2x over the simulator's serial on-demand execution (H100, batch 1; up to 3.7x on A100 / larger batches); with the whole 80 GB, 86-95% of expert accesses hit and
  only 0.8-2.3 ms of stall per token are left to hide. Prefetching attacks what remains: with a perfect predictor it
  is worth 1.06-1.19x at C = 224 and 1.30-1.37x when HBM is scarce (C = 64-128).
* **F1 - prediction.** Against the experts the simulator actually routed (top-2, 1 layer ahead): the training-free
  spec-hybrid predictor (calibrated speculative gate, history fallback for layer 0) reaches **0.943 recall / 0.895 full
  coverage** on TinyMoE prose (0.932 / 0.883 on code; the gate alone 0.925 / 0.888);
  history-only ST-MoE-style tables reach 0.64-0.67 (ST-MoE's own variable set: 0.71-0.73 recall with 3.5 experts).
  On routing with Mixtral's published, much weaker temporal locality the best top-2 recall is **0.61** (0.87 with
  top-4; 0.93 with the APEX-style adaptive budget at 4.8 experts). Uniform routing stays at the 0.25 chance level for every predictor.
* **F2/F3 - prefetching.** Batch 1, H100: on TinyMoE the prefetcher gives **1.27-1.28x at C = 64-128** (40-63% less
  stall, within 2-5% of the oracle) and **1.05-1.06x at C = 224** (69-73% less stall, within 1% of the oracle). On the
  Mixtral-calibrated routing it gives 1.04x / 1.09x / 1.11x / 1.08x at C = 64 / 128 / 192 / 224 (oracle 1.17-1.36x).
  On uniform routing it stays at 1.00x (within +-0.1%) - the feedback throttle all but switches it off - while the ST-MoE
  re-implementation loses up to 16% (20% on the A100 platform). Over all 54 platform x workload x batch x capacity
  points it is never slower than on-demand + LRU (worst 1.0004x); the history-only variant is at worst 0.5% slower.
  At batch 4-8 most experts are needed every layer and the link is saturated: <= 1.13x (oracle <= 1.17x).
  On the A100-class platform (80 GB HBM2e-class 2.04 TB/s, 450 GB/s link, batch 1) the picture is the same:
  1.20-1.25x on TinyMoE at C = 64-128, 1.06x at C = 224, 1.01-1.08x on Mixtral-calibrated, 1.00x on uniform.
* **F4 - energy and traffic** (H100, batch 1). Energy saving comes from background power during the shorter stalls;
  the cost is copy energy (13.4 pJ/bit) of extra traffic. With an accurate predictor and scarce HBM the saving wins
  (TinyMoE, C = 64-128: -9.7 ... -11.5% energy for +2 ... +8% traffic). With little stall left (C = 224: -1.7 ... -2.1%) or a
  weaker predictor (Mixtral-calibrated, C = 64-224: +1.4 ... -1.8%) the two roughly cancel while traffic grows 12-24%.
* **Validation.** In all 288 simulator runs (144 simulated batches x 2 modes) the simulator routed exactly the traced experts,
  and the timeline's cost model reproduces the simulator's expert latencies to 3e-14; the midterm D2 results
  (68.13 / 25.02 ms) reproduce on the modified simulator.

"ours" uses the TinyMoE / synthetic speculative-gate outputs where the workload has them; `uniform` has no hidden
states, so there it runs with the history-only ST-hybrid predictor. On `mixtral-calibrated` the speculative gate
is the generator's own conditional mean (an idealised predictor; see Limitations).

### Final deliverables -> where the evidence is

| Spec item | Evidence |
|---|---|
| F1. training-free predictor, accuracy / overlap vs the actually routed experts | `prefetch/predictors.py`; `experiments/f1_predictor_accuracy.py` -> `results/f1_accuracy.csv` (recall = overlap accuracy, precision, full coverage, set size for every predictor x lookahead x budget x batch), `results/f1_recall_per_layer.csv`, `results/f1_beta_sweep.csv`, `figures/f1_accuracy.png`. Ground truth = experts the simulator routed (`routing_log`, checked equal to the trace) |
| F2. prefetching mechanism into HBM | `prefetch/timeline.py` (HBM expert cache, link queue with demand priority, prefetch issue/promotion/cancellation, guided replacement, preemption, feedback throttling); ablation in `results/f2_summary.csv` (experiment = ablation) |
| F3. baseline vs prefetching: expert-load latency, total/token latency, exposed stalls, other metrics | `experiments/f2_prefetch.py` -> `results/f2_runs.jsonl.gz` (every run), `results/f2_summary.csv` (means, speed-ups), `results/f3_headline.csv`, `figures/f2_latency_capacity.png`, `figures/f2_breakdown.png`; on-demand baselines: framework (no reuse) and LRU cache |
| F4. energy, savings from reduced stalls, extra traffic | same files: `energy_*` columns split energy into op energy (simulator), DRAM->HBM copies, background x busy time, background x stall time; `traffic_gb_per_step`, `aborted_gb_per_step`, `prefetch_wasted`, `extra_traffic_vs_od_lru`; right-hand panels of `figures/f2_breakdown.png` |
| F5. code + README | this repository; `run_final.sh`; `tests/`; one representative simulator run in `experiments/sim_out/H100-NVL5_tinymoe-prose_B1_g0/` |
| Correlation analysis (midterm D4, extended) | `experiments/f0_correlation.py` -> `results/f0_correlation.csv`, `figures/f0_correlation.png` (vs Mixtral's published Table 5) |
| Validation | `experiments/validate_costs.py` -> `results/validate_costs.csv`; `tests/test_prefetch.py` |

(All paths under `experiments/` unless stated.) The written final report is the group's own work, as the
course requires; `RESULTS.md` only collects the numbers.

---

## 2. What was built for the final evaluation

### 2.1 Simulator modifications (SSD-offloading/LLMSimulator, C++)

The midterm showed that the simulator's own routing is a uniform random draw, so
"the actual routed experts" carry no temporal or spatial structure. The simulator was
extended (all edits marked `[MoE-prefetch project]`; diff in
`SSD-offloading/patch/moe_prefetch_routing_trace.patch`):

| File | Change |
|---|---|
| `src/scheduler/scheduler.{h,cpp}` | new optional config keys `simulation: routing_trace` (expert ids per batch slot / decode step / layer) and `simulation: routing_log`; trace loader with bounds checks |
| `src/scheduler/sequence.cpp` | `BatchedSequence::update_expert` takes the experts of every decode token from the trace (instead of `getRandomExpert`) and logs the experts it actually routed |
| `src/hardware/linear_impl.cpp`, `src/module/module_graph.cpp` | parse `config.yaml` once per process instead of once per operation (performance only; outputs unchanged) |
| `src/hardware/device.cpp` | accept `gpu_gen: A100` (the fork only knew H100/B100/B200): reuses the 80 GB HBM DRAM organisation, which only Ramulator uses (off in all runs); A100 bandwidth / FLOPs / capacity come from the existing `SystemConfig A100` |

Without the new keys the simulator behaves as before: the midterm D2 configs reproduce the
committed midterm CSVs (all columns identical except the unused `moe_comp` energy, 5e-8 relative,
floating-point summation order; checked by `experiments/validate_costs.py`). The trace loader rejects
out-of-range or repeated expert ids and traces shorter than the simulated batch / decode length.

### 2.2 Routing workloads (`workloads/`)

Real Mixtral-8x7B routing traces are not available offline and the 87 GB model cannot
run in this CPU-only setting, so four workloads bracket the behaviour:

| Workload | What it is | Why |
|---|---|---|
| `tinymoe-prose`, `tinymoe-code` | routing recorded from **TinyMoE**, a model with Mixtral's exact MoE topology (32 layers x 8 experts, top-2 softmax gating, SwiGLU experts, load-balancing loss) trained by us on Tiny Shakespeare / CPython stdlib code (`workloads/tinymoe/`) | decisions of a *learned* router on real text, including real hidden states for speculative gating |
| `mixtral-calibrated` | latent-logit generator whose consecutive-token repetition is fitted to the published Mixtral-8x7B statistics (Jiang et al. 2024, Table 5) | Mixtral's measured temporal locality, which is much weaker than TinyMoE's |
| `uniform` | the simulator's default routing | null hypothesis (nothing to predict) |

Each workload: 32 evaluation sequences x 256 tokens (first 64 = prompt history, 192
decode steps) plus a disjoint 32-sequence profiling split that is the only data used
to build count tables, calibrate or tune anything. F1 scores 16 sequences x 128 decode steps x 32 layers
(65,536 predictions per configuration at batch 1); F2-F4 simulate all 192 steps of 16 sequences at batch 1
(8 for the A100 platform and the ablation) and all 32 sequences at batch 4 and 8. Only the *routing pattern* comes from the
workload; all costs are Mixtral-8x7B on the simulated hardware.

### 2.3 Training-free predictors (`prefetch/predictors.py`)

All causal (a unit test perturbs not-yet-known routing and checks predictions do not change):

* **temporal** (ST-MoE History Table): previous token's experts at the target layer;
* **spatial** (ST-MoE Cross-layer Correlation Table): P(e at l+d | experts at l) from profiling counts;
* **ST-hybrid**: mix of the two, weight chosen on the profiling split (it selects the spatial table alone,
  lambda = 1, on every non-uniform workload - temporal evidence adds nothing once the previous layer is known);
* **ST-MoE re-implementation**: CCT with 2-bit confidence counters + HT, predict if score >= 2 (variable set size;
  the score also ranks resident experts when a prefetch needs a slot). Next layer only (d = 1);
* **spec-gate** (speculative gating, Eliseev & Mazur 2023): the target layer's own router applied to the
  hidden state available d layers earlier - training-free, re-uses existing router weights. Router softmax values
  are mapped to P(routed) with a histogram calibration (rank x softmax bin) counted on the profiling split;
* **spec-hybrid**: calibrated spec-gate, falling back to ST-hybrid for the first d layers of a token (no earlier
  hidden state of that token exists);
* frequency, random, oracle as references.

Budgets: fixed top-n, or **APEX-style adaptive** top-(k+delta): smallest set whose cumulative
predicted probability reaches tau.

### 2.4 Prefetcher and timeline simulator (`prefetch/timeline.py`)

A two-stream timeline (GPU compute stream + one DRAM->HBM copy link) driven by the simulator:

* HBM expert cache of C experts, partitioned per layer (Mixtral-offloading style), LRU;
* **on-demand** baselines: "no reuse" (every routed expert re-copied each step - the framework's copy volume; copies
  of one layer are pipelined with its expert execution, so latency is below the framework's serial load + exec sum)
  and on-demand + LRU cache;
* **prefetch** at ST-MoE timing (when layer l is routed, predict layer l+1; also d = 2 and APEX pre-attention d = 0 variants),
  with four mechanisms evaluated in the ablation:
  prediction-guided replacement (a prefetch only displaces a resident expert that is less likely and < 50% likely
  to be needed), preemptible prefetch (a demand miss aborts an in-flight copy at a projection boundary when that copy
  is known wrong or < 50% likely), accuracy-feedback throttling (feedback-directed prefetching, Srinath et al.
  HPCA'07), and the APEX-style adaptive budget;
* metrics: *prefetch accuracy* = share of prefetched experts routed at the layer they were predicted for;
  *eventual use* = executed at any time before eviction (the complement is wasted traffic);
* routing is never changed (correctness-preserving); demand fetches always overtake queued prefetches.

**Why a timeline layer on top of the simulator.** SSD-offloading measures expert loading and expert execution in
two separate passes (`collects_gpu_compute` off / on) and has no notion of HBM residency, a copy queue, or
overlap between copies and computation; LLMSimulator's module graph is built per iteration without a time-ordered
copy engine. A prefetcher is exactly the overlap + residency logic, so it is modelled as a discrete-event layer
(`prefetch/timeline.py`) driven by the simulator: the simulator decides the routing it executes (from the trace),
the attention/router time of every step and all per-op energies; the timeline adds when each expert copy happens
and whether the GPU has to wait. The simulator's own serial on-demand estimate (attention + all expert loads + all
expert executions, no overlap) is reported next to the timeline baselines (`sim_serial_on_demand_ms_per_step`).

Baselines: on-demand without reuse, on-demand + LRU, the ST-MoE predictor as published (variable set, its score
ranks residents for replacement) and the ST-MoE predictor combined with our mechanisms (separates predictor quality
from mechanism); upper bound: oracle prefetch.

**Design notes from the ablation** (`experiments/results/f2_summary.csv`, experiment = ablation; batch 1):
naive top-2 prefetch with LRU replacement already gives 1.25x on TinyMoE but *loses* on Mixtral-like routing at
C = 64 (0.98x, +35% traffic) because wrong prefetches evict experts that would have hit. Prediction-guided
replacement and preemption fix that (1.06x / 1.14x at C = 64 / 128). The feedback throttle gives up some of that
on Mixtral-calibrated (1.04x / 1.09x) in exchange for roughly half the extra traffic and, above all, for never
losing on unpredictable routing (uniform: 1.00x, whereas the unthrottled ST-MoE re-implementation drops to 0.84x);
a static confidence floor (A7) is slightly faster but needs a per-predictor threshold. The APEX-style adaptive budget changes nothing here: the calibrated speculative
gate is already confident on TinyMoE (2.1 experts at tau = 0.9, d = 1), and on Mixtral-like routing the extra experts it
asks for are throttled. Lookahead d = 2 helps TinyMoE slightly (1.30x at C = 128) and hurts Mixtral-like routing;
APEX-timed pre-attention issue (d = 0) is too late here, because one expert copy (391 us) is 10x longer than an
attention block.

Costs: attention/router time and per-op energies per step come from the simulator CSV rows of the same routing;
expert load/exec times use the SSD-offloading formulas re-implemented in `prefetch/costs.py`, which reproduce the
simulator's `expert_ffn` column to < 1e-12 relative error in both modes. Energy = simulator op energies + copy
energy (9.2 pJ/bit DDR5 read + 4.2 pJ/bit HBM write, framework constants) + background power x latency
(70 W GPU + 57 W DRAM, `SSD-offloading/LLMSimulator/figure_4_and_5.py`). The background term is how reduced stalls save energy; the copy term is
how extra prefetch traffic costs energy.

---

## 3. Repository layout

```
README.md, RESULTS.md                 this file; generated table of every final number
run_final.sh                          final pipeline (build -> experiments -> figures -> RESULTS.md)
requirements.txt                      Python packages; requirements-tinymoe.txt adds the TinyMoE training stack

configs/final_base.yaml               simulator config template for the final runs (1 GPU, no PIM, experts in DRAM)
prefetch/                             predictor + timeline package
  costs.py        SSD-offloading cost/energy formulas (validated)     workload.py   trace I/O
  predictors.py   training-free predictors + budgets                  timeline.py   prefetch / on-demand timeline
  evaluate.py     accuracy metrics + correlation statistics           simulator.py  drives the modified simulator
workloads/
  synthetic.py                        uniform + Mixtral-calibrated traces
  tinymoe/{fetch_corpus.sh,tinymoe.py,train_tinymoe.py,extract_traces.py}
  tinymoe/train_log.json              TinyMoE training curve (1000 steps; held-out CE 4.71 prose / 4.91 code nats)
  traces/tinymoe_{prose,code}.npz     TinyMoE routing workloads (committed, 9 MB); the two synthetic ones are
                                      regenerated deterministically by synthetic.py
experiments/
  f0_correlation.py                   F0  temporal/spatial correlation (extends midterm D4)
  f1_predictor_accuracy.py            F1  predictor accuracy / overlap vs routed experts
  f2_prefetch.py                      F2-F4 simulator runs + prefetch vs on-demand (resumable)
  validate_costs.py                   simulator routing == trace, cost model == simulator, midterm regression
  f3_plots.py, make_results_md.py     figures, headline table, RESULTS.md
  results/                            processed results (CSV; f2_runs.jsonl.gz = every timeline run)
  figures/                            final figures
  sim_out/H100-NVL5_tinymoe-prose_B1_g0/   one representative simulator run (config, trace, logs, CSVs)
tests/test_prefetch.py                unit tests (closed-form timeline cases, causality, I/O)

SSD-offloading/                       simulator used for D2 and the final evaluation (modified, see 2.1)
LLMSimulator/                         simulator used for D1 / D2-OOM / D3 (upstream 4192527 + PIM patch)
mixtral_*.yaml, ssd_cfgs/, analysis_d*.py, run_mixtral_*.log, ssd_results/, d4_correlation.png
                                      midterm configs, scripts and outputs (section 5)
rerun_backup/                         midterm D1 isolation runs (TP4 alone, Duplex without parallel execution)
report/                               midterm report
```

---

## 4. Reproducing the final evaluation

### 4.1 Prerequisites

* Linux (or a Linux container, e.g. `ubuntu:22.04`/`24.04`) with `g++` (11 or newer), `cmake` >= 3.14, `make`, `git`
  (CMake fetches `yaml-cpp` 0.7.0 from GitHub at configure time; with CMake >= 4.0 the extra flag
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` used below is required, older CMake ignores it).
* Python 3.12+ with the packages in `requirements.txt` (`numpy`, `matplotlib`, `PyYAML`; tested with Python 3.13;
  the pinned numpy needs >= 3.12).
  Only for regenerating the TinyMoE traces: `pip install -r requirements-tinymoe.txt` (`jax[cpu]`, `optax`,
  `tokenizers`) and network access to `raw.githubusercontent.com`.

```bash
git clone https://github.com/breaking-byts/MoE-Memory-Wall-CA-Project.git
cd MoE-Memory-Wall-CA-Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 One command

```bash
./run_final.sh            # uses the committed routing traces
./run_final.sh --retrain  # additionally re-downloads the corpus, retrains TinyMoE (~45 min) and re-extracts traces
```

Measured on a 2-core container: simulator build ~5 min, F1 ~5 min, F2-F4 ~70 min single-process
(144 simulated batches x 2 simulator modes + 5,472 timeline runs; resumable - rerunning skips finished
runs and cached simulator outputs), rest < 2 min.

### 4.3 Step by step (what `run_final.sh` does)

```bash
# build the modified simulator (-include cstdint is needed by ramulator2 with g++ >= 13; harmless on g++-11)
cd SSD-offloading/LLMSimulator && mkdir -p build && cd build
cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-include cstdint" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. \
  && make -j"$(nproc)" run
cd ../../..

# midterm D2 regression on the modified simulator (no routing trace; read by validate_costs.py)
ROOT=$PWD; REG=experiments/sim_out/midterm_regression; SIM=SSD-offloading/LLMSimulator
mkdir -p $REG && rm -f $SIM/build/tmp/*.csv
for c in cpu cpu_copy; do
  cp ssd_cfgs/$c.yaml $SIM/build/config.yaml && (cd $SIM/build && ./run > "$ROOT/$REG/ssd_$c.log")
done
cp $SIM/build/tmp/*.csv $REG/

# routing workloads: synthetic ones are regenerated (deterministic); TinyMoE ones are committed
python3 workloads/synthetic.py
python3 tests/test_prefetch.py
python3 experiments/f0_correlation.py
python3 experiments/f1_predictor_accuracy.py
python3 experiments/f2_prefetch.py        # runs ./run in SSD-offloading/LLMSimulator/build for every batch
python3 experiments/validate_costs.py     # exits non-zero if any check fails
python3 experiments/f3_plots.py
python3 experiments/make_results_md.py
```

Running one simulator configuration by hand (the framework reads `./config.yaml` and
`./dram_config_HBM3_80GB.yaml` from its working directory; all paths in the config are relative to it):

```bash
cd experiments/sim_out/H100-NVL5_tinymoe-prose_B1_g0/work_compute     # or work_copy
cp ../../../../SSD-offloading/LLMSimulator/dram_config_HBM3_80GB.yaml .
../../../../SSD-offloading/LLMSimulator/build/run    # rewrites ../csv_compute/*.csv and ../routing_log_compute.csv
rm -rf dram_config_HBM3_80GB.yaml log; cd -
```

### 4.4 External data

No dataset is committed. `workloads/tinymoe/fetch_corpus.sh` downloads (only needed with `--retrain`):

* `input.txt` (Tiny Shakespeare, 1,115,394 bytes; saved as `corpus/prose.txt`) from
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
* 58 CPython **v3.12.0** standard-library modules (listed in the script) from
  `https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/<module>.py` (2,984,196 bytes total; concatenated
  into `corpus/code.txt`)

The TinyMoE checkpoint is not committed; the extracted routing traces (`workloads/traces/tinymoe_*.npz`) are.

### 4.5 Verifying a run

* `python3 experiments/validate_costs.py` prints `PASS` (all simulator runs routed exactly the traced experts; cost
  model matches the simulator; midterm CSVs reproduced). It checks, and rewrites `results/validate_costs.csv` for,
  whatever simulator runs exist under `experiments/sim_out/`.
* F2 is resumable: `experiments/results/f2_runs.jsonl` (not committed; a gzip copy is) collects finished runs and
  `f2_prefetch.py` skips them; delete it to recompute everything. Cached simulator outputs in `experiments/sim_out/`
  are reused the same way - including the committed representative run `H100-NVL5_tinymoe-prose_B1_g0` (delete its
  `done.json` to re-simulate it; the outputs are byte-identical). `f3_plots.py` and `make_results_md.py` read
  `results/f2_summary.csv`, which `python3 experiments/f2_prefetch.py` writes at the end; after a partial or manual
  F2 run regenerate it with `python3 experiments/f2_prefetch.py --summary-only`.
* `python3 tests/test_prefetch.py` prints `ok` for every test.
* Compare `RESULTS.md` with the committed one (timeline and simulator are deterministic).

---

## 5. Midterm (D1-D4) - unchanged

| # | Result |
|---|---|
| D1 | Reproduced Duplex (GPU+Logic-PIM, TP4): **3.10x faster decode**, 23.94 ms -> 7.73 ms on 4xH100, batch 32 |
| D2 | Mixtral does not fit one 80 GiB GPU (**88.48 > 80 GiB**); experts in off-chip DRAM: **loading 68.13 ms vs execution 25.02 ms = 2.72x** |
| D3 | Expert FFNs are **96.2%** of decoder time; one projection's weight read costs **98x** its arithmetic; mean **7.28 of 8** experts active per layer |
| D4 | Simulator routing has **no** temporal/spatial correlation (overlaps sit on the 0.25 null); a synthetic ST-MoE-like trace reaches 0.56/0.83 overlap |

Reproduce (LLMSimulator for D1/D3, SSD-offloading for D2; use the same build flags as 4.3 for LLMSimulator):

```bash
cd LLMSimulator && mkdir -p build data && cd build && \
  cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-include cstdint" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. \
  && make -j"$(nproc)"
./run ../../mixtral_gpu.yaml        > ../../run_mixtral_gpu.log
./run ../../mixtral_duplex.yaml     > ../../run_mixtral_duplex.log
./run ../../mixtral_single_gpu.yaml > ../../run_mixtral_single.log
cd ../..
# D2 (SSD-offloading reads ./config.yaml)
cd SSD-offloading/LLMSimulator/build
cp ../../../ssd_cfgs/cpu.yaml config.yaml      && ./run > ../../../ssd_results/ssd_cpu.log
cp ../../../ssd_cfgs/cpu_copy.yaml config.yaml && ./run > ../../../ssd_results/ssd_cpu_copy.log
cp tmp/*.csv ../../../ssd_results/
cd ../../..
python3 analysis_d2.py; python3 analysis_d3.py; python3 analysis_d4.py
grep -h "^Total:" run_mixtral_gpu.log run_mixtral_duplex.log   # 23943455.249045 / 7725699.519236 ns
```

Quirks: the SSD framework reads `./config.yaml` from its working directory; `iter: n` writes n-1 CSV rows
(row k = decode step k); in copy mode only the `expert_ffn` column is meaningful; re-runs are bit-identical
except one unused energy column (`moe_comp`, 5e-8 relative, floating-point summation).

---

## 6. Limitations

* Both simulators are analytical (roofline, no DRAM timing); copy/GPU contention is the framework's fixed
  bandwidth subtraction (HBM bandwidth minus link bandwidth during GPU work).
* TinyMoE is a small model (d = 64) trained on ~1 M tokens: its routing is genuinely learned but its
  locality (and especially the ~0.94 speculative-gating recall) is likely stronger than Mixtral's, and its
  per-layer expert popularity is skewed (`figures/f0_correlation.png`), which also helps the frequency /
  history predictors. `mixtral-calibrated` matches Mixtral's published temporal statistics, but its cross-layer
  strength is an assumption (beta, swept in F1), and its "speculative gate" is the generator's own conditional
  mean, i.e. an idealised (Bayes-optimal) predictor for that generator - an upper bound for speculative gating
  there, not a real router. The two workloads bracket the expected behaviour; neither is a Mixtral measurement.
* The simulator's attention cost uses a 512-token context (as in the midterm); the 64-token prompt of the traces
  only provides routing history to the predictors. 192 decode steps are traced and simulated; the simulator writes
  one CSV row per step except the last (`iter: n` -> n-1 rows), so 191 steps are evaluated.
* One copy engine, no compression/quantisation of experts, decode only (prefill streams all experts once).
* `A100-NVL4` is a what-if platform: an A100-class 80 GB HBM (the spec's reference local memory) behind the
  simulator's slower link option (450 GB/s); real A100 hosts use PCIe. Background power keeps the framework's
  H100 value (70 W + 57 W).
* Energy: framework per-op energies + background power; GPU active vs idle power is not distinguished.
* Per-step latency covers the 32 decoder layers (attention, router, experts), like the framework's own latency
  accounting; embedding and LM head are excluded.

## 7. References

1. S. Yun et al., "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching," MICRO 2024, arXiv:2409.01141.
2. Y. Zhao et al., "A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference," arXiv:2606.15453, 2026 (ST-MoE).
3. A. Kanani, L. Badawi, U. Y. Ogras, "APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE Inference," arXiv:2608.11688, 2026.
4. A. Q. Jiang et al., "Mixtral of Experts," arXiv:2401.04088, 2024 (Table 5 routing statistics).
5. A. Eliseev, D. Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading," arXiv:2312.17238, 2023 (speculative gating, per-layer LRU).
6. S. Srinath, O. Mutlu, H. Kim, Y. Patt, "Feedback Directed Prefetching," HPCA 2007.
7. LLMSimulator https://github.com/scale-snu/LLMSimulator ; SSD-offloading https://github.com/scale-snu/SSD-offloading
