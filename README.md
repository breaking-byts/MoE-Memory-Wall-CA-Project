# The MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

Course project (CA Project 2): study the Mixture-of-Experts memory wall for LLM
inference using the Mixtral-8x7B model, with the LLMSimulator and SSD-offloading
simulation frameworks. This repo holds the **mid-semester submission** (deliverables
D1–D4). The full write-up is [`report/CA Midterm Report Group 24.pdf`](<report/CA Midterm Report Group 24.pdf>).

## Headline results

| # | Result |
|---|---|
| D1 | Reproduced Duplex (GPU+Logic-PIM, TP4): **3.10× faster decode**, 23.94 ms → 7.73 ms on 4×H100, batch 32 |
| D2 | Mixtral does not fit one 80 GiB GPU (**88.48 > 80 GiB**); experts in off-chip DRAM: **loading 68.13 ms vs execution 25.02 ms = 2.72×** |
| D3 | Expert FFNs are **96.2%** of decoder time; one projection's weight read costs **98×** its arithmetic; mean **7.28 of 8** experts active per layer |
| D4 | Simulator routing has **no** temporal/spatial correlation (overlaps sit on the 0.25 null); a synthetic ST-MoE-like trace reaches 0.56/0.83 overlap |

## Layout

```
mixtral_gpu.yaml / mixtral_duplex.yaml      D1 configs (baseline vs Duplex, 4×H100)
mixtral_single_gpu.yaml                     D2 single-GPU config (also feeds D3's op tree)
ssd_cfgs/cpu.yaml / cpu_copy.yaml           D2 execution / loading configs (1 GPU, experts in CPU DRAM)
LLMSimulator/                               simulator for D1/D2-OOM/D3 (upstream 4192527 + PIM patch)
SSD-offloading/                             offload-model fork for D2 load-vs-exec
analysis_d2.py / analysis_d3.py             OOM + load/exec and MoE-flow analyses (stdlib only)
analysis_d4.py -> d4_correlation.png        routing-correlation statistics and plots (needs numpy+matplotlib)
run_mixtral_*.log / ssd_results/            raw simulator outputs the analyses read
rerun_backup/                               verification evidence (fresh rebuild + isolation runs)
report/                                     final midterm report PDF + D4 figure
```

## Prerequisites

- A Linux container host. The team used Podman with `ubuntu:22.04`
  (`podman machine init && podman machine start` first on macOS, then the
  commands below; Docker works the same with `docker` in place of `podman`).
  Inside the container: `g++-11 gcc-11 cmake make git`
  (`git` is required: CMake fetches `yaml-cpp` at configure time).
- Python 3 with pip and venv (`analysis_d2.py` and `analysis_d3.py` use the
  standard library only; only `analysis_d4.py` needs `requirements.txt`).
- No physical GPU needed: everything here is simulation.
- Timings on a 4-core container host: clone ~5 s, pip install ~10 s,
  toolchain install ~1 min, LLMSimulator build ~7 min, SSD-offloading
  build ~4 min, all simulation runs < 10 s, analyses ~10 s (~12 min total).

## Reproduce (from a fresh clone, without any checked-in build)

```bash
# 0. Clone and enter
git clone https://github.com/breaking-byts/MoE-Memory-Wall-CA-Project.git
cd MoE-Memory-Wall-CA-Project

# 1. Python env (only analysis_d4.py needs it)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build (inside the container; LLMSimulator needs data/ to exist for CSV export)
podman run -dit --name rerun -v "$PWD":/work docker.io/library/ubuntu:22.04 sleep infinity
podman exec rerun bash -lc "apt-get update && apt-get install -y g++-11 gcc-11 cmake make git"
podman exec rerun bash -lc "cd /work/LLMSimulator && mkdir -p build data && cd build && \
  cmake -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_STANDARD=17 .. && make -j4"
podman exec rerun bash -lc "cd /work/SSD-offloading/LLMSimulator && mkdir -p build && cd build && \
  cmake -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_STANDARD=17 .. && make -j4 run"

# 3. Run the simulations
podman exec rerun bash -lc "cd /work/LLMSimulator/build && \
  ./run ../../mixtral_gpu.yaml        > ../../run_mixtral_gpu.log && \
  ./run ../../mixtral_duplex.yaml     > ../../run_mixtral_duplex.log && \
  ./run ../../mixtral_single_gpu.yaml > ../../run_mixtral_single.log"
# D2 load vs exec (the SSD framework reads ./config.yaml from the working directory)
podman exec rerun bash -lc "cd /work/SSD-offloading/LLMSimulator/build && \
  cp ../../../ssd_cfgs/cpu.yaml config.yaml      && ./run > ../../../ssd_results/ssd_cpu.log && \
  cp ../../../ssd_cfgs/cpu_copy.yaml config.yaml && ./run > ../../../ssd_results/ssd_cpu_copy.log && \
  cp tmp/*.csv ../../../ssd_results/"

# D1 isolation: the same D1 configs with one field flipped each
# (mixtral_gpu.yaml with both tensor degrees 4; mixtral_duplex.yaml with
# parallel_execution off) — evidence kept in rerun_backup/

# 4. Analyse (on the host, venv activated)
python3 analysis_d2.py
python3 analysis_d3.py
python3 analysis_d4.py

# 5. Verify your outputs match the committed ones
grep -h "^Total:" run_mixtral_gpu.log run_mixtral_duplex.log
# expect 23943455.249045 and 7725699.519236 (ns)
grep "^Total:" run_mixtral_single.log
# expect 88.4792GB footprint and 75245619.655333 (ns)
```

Expected outputs: D1 totals 23.943 ms / 7.726 ms (3.10×); D2 OOM 88.48 GiB,
load 68.13 ms / exec 25.02 ms (2.72×); D3 96.2% expert share, 98× roofline replay;
D4 statistics table + `d4_correlation.png`. All three scripts exit 0.

## Reproducibility notes (quirks)

- The SSD framework reads `./config.yaml` from its working directory — copy the
  config into the build dir; passing a path on the command line does nothing.
- `iter: 3` writes **two** CSV rows (export runs *before* each iteration, so the
  last iteration is never written); row 0 is always used.
- With `use_ramulator: off` both simulators are pure roofline arithmetic, so
  re-runs are bit-identical (verified: logs byte-identical, one unused energy
  column drifts by 5e-8 from FP summation).
- In copy mode the framework also charges a copy for attention weights, so only
  the `expert_ffn` column is meaningful; in compute mode GPU execution sees
  HBM bandwidth minus the copy link (3352 − 900 = 2452 GB/s).
- D1 isolation runs (TP4-alone, Duplex without parallel execution) live in
  `rerun_backup/`; they show the 3.10× is essentially all Logic-PIM.

## References

- S. Yun et al., "Duplex: A Device for Large Language Models with Mixture of
  Experts, Grouped Query Attention, and Continuous Batching," arXiv:2409.01141, 2024
  (copy: `2409.01141v1.pdf`).
- Y. Zhao et al., "A Spatio-Temporal Expert Prefetching Framework for Efficient
  MoE-based LLM Inference," arXiv:2606.15453, 2026 (ST-MoE).
- A. Kanani et al., "APEX: Adaptive Expert Prefetching for Memory-Efficient Edge
  MoE Inference," arXiv:2608.11688, 2026.
- A. Q. Jiang et al., "Mixtral of Experts," arXiv:2401.04088, 2024.
- LLMSimulator: https://github.com/scale-snu/LLMSimulator ·
  SSD-offloading: https://github.com/scale-snu/SSD-offloading
