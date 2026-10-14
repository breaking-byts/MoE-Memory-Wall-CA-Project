# Single-GPU Offload-Tier Model — Midterm Robustness Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the evaluator's three D2 gaps (4-device configs, C2C-only offload path, no comparable load+exec measurement) and the D3 gap (per-op example uses DeepSeek, not Mixtral), then re-verify and debloat the whole codebase for midterm.

**Architecture:** A new project patch (`SSD-offloading/patch/offload_tier.patch`) layered on the vendored `ssd_offloading.patch` adds (a) an explicit off-chip tier bandwidth behind the C2C link and (b) mode-independent load-time/energy counters that surface as a new `expert_load` CSV column, so one run reports load and execution comparably. All 5 offload configs move to `num_device: 1`. The D3 showcase moves to a Mixtral per-op trace produced by flipping `print_log: on`.

**Tech Stack:** C++17 (vendored LLMSimulator/SSD-offloading, built in podman `ubuntu:22.04`), Python 3 (analysis scripts, `./.venv`), podman VM `ca-vm` (already repaired and running).

**Context notes:**
- CA_Project root is NOT a git repo — no worktree, no commits. The vendored `SSD-offloading/LLMSimulator` IS a git repo whose working tree already carries the applied `ssd_offloading.patch` (uncommitted). Our delta is isolated via the **git index trick**: `git add -A` BEFORE editing (stages the ssd-patched baseline), then after edits `git diff` = exactly our patch. `git reset` at the very end (no commits ever made).
- Run everything through the existing podman pattern (MIDTERM_GUIDE §2):
  `podman run --rm -v "$PWD:/work:z" -w /work/... docker.io/library/ubuntu:22.04 bash -c '...'`
  Use SINGLE QUOTES around inner scripts (guide's quoting trap).
- The SSD-offloading binary reads offload flags from `build/config.yaml` in CWD (ignores argv) — every run must `cp ssd_cfgs/<m>.yaml SSD-offloading/LLMSimulator/build/config.yaml` first.

---

### Task 0: Preflight — isolate the vendored repo state

**Step 1:** Stage the current (ssd-patched) state so later `git diff` shows only our edits:

```bash
git -C SSD-offloading/LLMSimulator add -A
git -C SSD-offloading/LLMSimulator status --short | head -5   # expect: staged M entries, no ?? untracked (build/ is ignored)
```

### Task 1: C++ — ExecStatus/StatusBoard fields

**Files:**
- Modify: `SSD-offloading/LLMSimulator/src/module/status.h`

**Step 1:** In `struct ExecStatus`, after `time_ns memory_duration = 0.0;` add:

```cpp
  time_ns offload_duration = 0.0;  // time to copy weight over the offload tier (independent of collects_gpu_compute)
```

**Step 2:** In `class StatusBoard`, after `time_ns low_time = 0;  // time of PIM` add:

```cpp
  time_ns offload_time = 0;  // accumulated expert-weight copy time over the offload tier
```

### Task 2: C++ — explicit tier bandwidth in SystemConfig

**Files:**
- Modify: `SSD-offloading/LLMSimulator/src/hardware/hardware_config.h`

**Step 1:** After the member `hw_metric offload_bandwidth;` (~line 126) add:

```cpp
  hw_metric offload_tier_bandwidth = 0.0;      // bytes/s of the off-chip tier behind the link (0 = link-only)
  hw_metric effective_offload_bandwidth = 0.0; // min(C2C link, tier): sustained copy rate
```

**Step 2:** In the constructor body, change the contention line (~line 97) from

```cpp
          memory_bandwidth = base_memory_bandwidth - offload_bandwidth;
```

to

```cpp
          memory_bandwidth = base_memory_bandwidth - effective_offload_bandwidth;
```

(A tier-limited copy cannot steal more HBM bandwidth than it actually transfers.)

**Step 3:** In `parseOffloadConfig()`, extend the `if (offload_cpu || offload_ssd)` branch after the `gen` selection:

```cpp
        // Explicit off-chip tier behind the C2C link, in GB/s. 0 or absent =
        // no tier limit (copy runs at full link bandwidth -- the previous model).
        double tier = 0.0;
        if (config["system"]["offload"]["offload_tier_bandwidth"]) {
          tier = config["system"]["offload"]["offload_tier_bandwidth"].as<double>() * 1.0e9;
        }
        offload_tier_bandwidth = tier;
        effective_offload_bandwidth =
            (tier > 0.0) ? std::min(offload_bandwidth, tier) : offload_bandwidth;
```

and in the `else` branch add:

```cpp
        offload_tier_bandwidth = 0.0;
        effective_offload_bandwidth = 0.0;
```

**Step 4:** Ensure `#include <algorithm>` is present in `hardware_config.h` (for `std::min`).

### Task 3: C++ — linear_impl latency + energy forks

**Files:**
- Modify: `SSD-offloading/LLMSimulator/src/hardware/linear_impl.cpp`

**Step 1:** In `LinearExecutionGPU`, inside the offload `if` (~lines 51-65), compute the copy time once and record it; replace the whole block with:

```cpp
  if (set_config["system"]["offload"]["offload_expert_to_cpu_memory"].as<bool>() ||
      set_config["system"]["offload"]["offload_expert_to_ssd"].as<bool>()) {
      // Copy time of the weight over the offload tier. Recorded in BOTH modes so
      // a single run reports load (offload_duration -> CSV expert_load) and
      // execution separately; effective_offload_bandwidth = min(C2C link, tier).
      time_ns copy_duration =
          (k * n) * weight->precision_byte / config.effective_offload_bandwidth * 1000 * 1000 * 1000;
      exec_status.offload_duration = copy_duration;
      if (set_config["system"]["offload"]["collects_gpu_compute"].as<bool>()) { // collects GPU-compute latency
        time_ns memory_duration =
          total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        exec_status.memory_duration = memory_duration;
        exec_status.compute_duration = compute_duration;
        exec_status.total_duration =
          std::max(exec_status.compute_duration, exec_status.memory_duration);
      } else { // collects offloaded-weight copying latency
        exec_status.total_duration = copy_duration;
      }
  }
```

**Step 2:** In the energy section (~lines 91-104), make the copy-energy counter mode-independent; replace the block with:

```cpp
  if (set_config["system"]["offload"]["offload_expert_to_cpu_memory"].as<bool>() ||
      set_config["system"]["offload"]["offload_expert_to_ssd"].as<bool>()) {
    // The copy over the offload tier happens in both modes -- count its energy
    // in the same run as GPU-compute energy (moe_offload was previously
    // structurally 0 when collects_gpu_compute was on).
    exec_status.moe_offload_read_count = k * n * weight->precision_byte;
    if (set_config["system"]["offload"]["collects_gpu_compute"].as<bool>()) {
      // weights already staged in HBM: compute reads them from HBM
      exec_status.moe_hbm_read_count = traffic[0];
    }
  } else {
    // MoE without offloading (baseline)
    exec_status.moe_hbm_read_count = traffic[0];
  }
```

### Task 4: C++ — aggregate `offload_time` and export `expert_load`

**Files:**
- Modify: `SSD-offloading/LLMSimulator/src/module/module_graph.cpp`
- Modify: `SSD-offloading/LLMSimulator/src/module/timeboard.h`
- Modify: `SSD-offloading/LLMSimulator/src/module/timeboard.cpp`
- Modify: `SSD-offloading/LLMSimulator/src/hardware/stat.h`
- Modify: `SSD-offloading/LLMSimulator/src/hardware/cluster.cpp`

**Step 1:** `module_graph.cpp`: after EVERY `status.device_time += exec_status.total_duration;` add:

```cpp
    status.offload_time += exec_status.offload_duration;
```

(Find all sites with `grep -n "device_time += exec_status.total_duration"`. `offload_duration` is 0 for non-offloaded ops, so unconditional accumulation is safe.)

**Step 2:** `timeboard.h` in `class TimeStamp`, next to `getMoeOffloadEnergy()` add:

```cpp
  time_ns getOffloadTime(){
    return status.offload_time;
  }
```

**Step 3:** `timeboard.cpp` in `set_status(...)`, next to the moe energy deltas add:

```cpp
  status.offload_time = output_status.offload_time - status.offload_time;
```

**Step 4:** `stat.h` after `time_ns expert_ffn = 0;` add:

```cpp
  time_ns expert_load = 0;
```

**Step 5:** `cluster.cpp` — `setTimeBreakDown` has TWO parallel branches (stamps found ~line 679 and ~line 903; expertFFN loops ~787 and ~1077). In BOTH:
- declare `time_ns expert_load = 0;` next to `time_ns expert_ffn = 0;`
- in the `for (auto stamp : ExpertFFN) { expert_ffn += ...}` loop add `expert_load += stamp->getOffloadTime();`
- in the `for (auto stamp : SharedExpertFFN)` loop accumulate a `shared_expert_load` the same way (from SharedExpertFFN stamps)
- set `stat.expert_load = expert_load - shared_expert_load;` next to `stat.expert_ffn = ...` (gated-experts-only, mirroring expert_ffn's subtraction; Mixtral has no shared expert so this is 0)

**Step 6:** `cluster.cpp` `exportToCSV` (~line 349): after the `expert_ffn` line insert:

```cpp
        << std::to_string(temp.expert_load) << "," // expert weight copy over offload tier
```

**Step 7:** `cluster.cpp` `runIteration` header (~line 398): in the header string, after `expert_ffn,` insert `expert_load,` (both the `runIteration` copy at ~:398 and the second header in `runIterationMixed`/`runIterationSumGenSplit` if present — grep `o_proj,ffn,shared_expert_ffn,expert_ffn,communication` to find all header strings and update all).

### Task 5: Build

```bash
podman run --rm -v "$PWD:/work:z" -w /work/SSD-offloading/LLMSimulator/build \
  docker.io/library/ubuntu:22.04 bash -c \
  "apt update -qq && apt install -y -qq g++-11 cmake clang git python3 >/dev/null 2>&1; make -j4 2>&1 | tail -5"
```

Expected: `[ ... ] Built target run` (or no errors; incremental). If compile errors → fix and re-run.

### Task 6: Single-GPU configs

**Files:**
- Modify: `ssd_cfgs/hbm.yaml`, `cpu.yaml`, `cpu_copy.yaml`, `ssd.yaml`, `ssd_copy.yaml`

In each of the 5: `num_device: 4` → `1`; `expert_tensor_degree: 1` (already 1); `none_expert_tensor_degree: 4` → `1`.
In `cpu.yaml`/`cpu_copy.yaml` add under `offload:`:

```yaml
    offload_tier_bandwidth: 32 # GB/s CPU DDR5 server DIMM behind the NVLink-C2C link
```

In `ssd.yaml`/`ssd_copy.yaml` add:

```yaml
    offload_tier_bandwidth: 7 # GB/s M.2 NVMe SSD behind the NVLink-C2C link
```

In `hbm.yaml` add under `system:` (single GPU cannot hold all expert weights — this is the motivation result, keep the run alive):

```yaml
  exit_out_of_memory: off
```

If `exit_out_of_memory` turns out to be unsupported by the SSD-offloading config parser (run aborts on unknown key or on OOM), adapt: verify empirically in Task 7 and record actual behavior in the guide.

### Task 7: Re-run all 5 modes + collect artifacts

```bash
mkdir -p ssd_results/4device_archive
mv ssd_results/mixtral_synthesis_N1_D4_*.csv ssd_results/4device_archive/
for m in hbm cpu cpu_copy ssd ssd_copy; do
  cp "ssd_cfgs/$m.yaml" SSD-offloading/LLMSimulator/build/config.yaml
  podman run --rm -v "$PWD:/work:z" -w /work/SSD-offloading/LLMSimulator/build \
    docker.io/library/ubuntu:22.04 bash -c "./run > /work/ssd_results/ssd_${m}.log 2>&1; echo EXIT=\$?"
  cp "ssd_cfgs/$m.yaml" "ssd_results/$m.yaml"
done
cp SSD-offloading/LLMSimulator/build/tmp/*.csv ssd_results/
ls ssd_results/
```

Expected: 5 EXIT=0; new CSVs named `mixtral_synthesis_N1_D1_TP1_DP1_batchsize4_<mode>.csv`; each CSV row has `expert_load` right after `expert_ffn`. Check `hbm.yaml` row for `OOM=1` (expected: all-resident single GPU exceeds 80 GiB — the motivation). If a run aborts on OOM, add/adjust `exit_out_of_memory` handling per Task 6 note.

### Task 8: Generate the project patch + wire into patches.sh

**Step 1:**

```bash
git -C SSD-offloading/LLMSimulator diff > SSD-offloading/patch/offload_tier.patch
wc -l SSD-offloading/patch/offload_tier.patch     # expect a few hundred lines
git -C SSD-offloading/LLMSimulator apply --check --reverse SSD-offloading/patch/offload_tier.patch && echo "patch matches working tree"
```

**Step 2:** Append to `SSD-offloading/patches.sh` (before the final echo):

```bash
echo "▶︎  Applying project patch (explicit off-chip tier + one-run load/exec reporting)..."
pushd "$SIM_DIR" >/dev/null
git apply --whitespace=nowarn "$ROOT_DIR/patch/offload_tier.patch"
popd >/dev/null
```

### Task 9: analysis_d2.py rework

**Files:**
- Modify: `analysis_d2.py`

1. **Constants**: add `CPU_TIER_GBPS = 32e9`, `SSD_TIER_GBPS = 7e9` (with source comment "ssd_cfgs/*.yaml offload_tier_bandwidth"); replace the `CPU_DDR_BW_GBPS`/`SSD_BW_GBPS` literature constants usage in `table_scaling` (rows now simulated; update the note text accordingly — CPU/SSD rows become "sim models min(link, tier)").
2. Remove the dead `exec_ms` variable in `table_scaling`.
3. **`table_ssd_results`**: read `expert_load` too; print a new comparable line from ONE run (CPU GPU-compute): `expert_load` vs `expert_ffn` from the same CSV row.
4. **`contention_model`**: replace hardcoded 6.52 ms with the measured baseline `expert_ffn` read from the hbm CSV (glob `*baseline*GPU_compute.csv`).
5. **`check()`** updates:
   - Replace "CPU and SSD copy latency identical" with "CPU/SSD copy-load ratio = tier ratio 32/7": `_f(ssd,'expert_load')/_f(cpu,'expert_load') ≈ 32/7` (tolerance 1%).
   - Keep energy-ratio and bytes-moved checks (values re-read from new CSVs automatically).
   - Contention check: predicted ratio now `HBM/(HBM − 32 GB/s) = 3.352/3.320 = 1.00964` — parameterize from the tier constant.
   - NEW: mode-independence — `expert_load` from `*_GPU_compute.csv` equals `*_offloaded_weight_copy.csv` within 0.1% for the same tier.
   - NEW: analytic load time — back-solve bytes from `moe_offload` energy, predict `bytes / min(900 GB/s, tier)` and compare to CSV `expert_load` (tolerance 1%).
6. Update the printed caveat texts to the new model (no more "structurally 0", C2C-only note becomes tier note).

**Verify:** `./.venv/bin/python analysis_d2.py --check` → all checks pass. (Do NOT update expected numbers in the guide yet — that is Task 11 with real values.)

### Task 10: D3 Mixtral per-op trace

**Files:**
- Modify: `mixtral_synth64.yaml` (`print_log: off` → `on`)
- Create: `run_synth64.log` (replaces the old tiny one; same run + per-op printing)

```bash
cp LLMSimulator/data/mixtral_synthesis_64_16_GPU_N1_D1_TP1_DP1_maxbatch8_maxprocess524288_iter3_skew0_precision_byte2_parallel_execution0.csv /tmp/synth64_before.csv
sed -i '' 's/  print_log: off/  print_log: on/' mixtral_synth64.yaml
podman run --rm -v "$PWD:/work:z" -w /work/LLMSimulator/build docker.io/library/ubuntu:22.04 \
  bash -c './run /work/mixtral_synth64.yaml > /work/run_synth64.log 2>&1; echo EXIT=$?'
diff /tmp/synth64_before.csv LLMSimulator/data/mixtral_synthesis_64_16_*.csv && echo "CSV identical: print_log does not affect results"
./.venv/bin/python analysis_d3.py run_synth64.log
```

Expected: EXIT=0; CSV byte-identical; analysis_d3 detects `mixtral` (H100 constants) and produces the per-op distribution + roofline replay for Mixtral. Keep the output for Task 11.

### Task 11: Guide updates (MIDTERM_GUIDE.md)

1. Remove the duplicated §7d block (the second "7d. Trace-driven sim" section, ~lines 518-528).
2. §2: add the podman repair note (vfkit "no such file ca-vm-arm64.raw" → `podman machine rm ca-vm --force && podman machine init --cpus 4 --memory 8192 --disk-size 40 ca-vm && podman machine start ca-vm`).
3. §3: document the new `patch/offload_tier.patch` + the patches.sh block.
4. §5 rewrite (D2): single-GPU configs (`N1_D1_TP1_DP1`), explicit tier model (`offload_tier_bandwidth`, effective BW = min(link, tier), contention uses effective BW), one-run load+exec (`expert_ffn` + `expert_load`), hbm-baseline OOM=1 on one GPU (motivation), updated tables/checks with the REAL numbers from Tasks 7-9, D2a tier table rows updated (CPU/SSD rows now simulator settings).
5. §6c: Mixtral per-op example from `run_synth64.log` (roofline replay + load:compute ratio from Task 10 output); keep DeepSeek as secondary with the model-identity warning.
6. §8 inventory: new files (`patch/offload_tier.patch`, `ssd_results/4device_archive/`, new CSV names, `run_synth64.log`), remove references to deleted/renamed artifacts if any.

### Task 12: Debloat sweep

1. `analysis_d4.py`: delete the unused `phases()` function (verify first: `grep -rn "phases(" --include="*.py" .` shows only its definition).
2. `test_trace_io.py`: remove the pointless `out`/`names`/`del` lines in `test_persistence_is_reported_for_the_decode_phase`.
3. Clean stale `SSD-offloading/LLMSimulator/build/tmp/` CSVs after results are copied (regenerable runtime artifacts).
4. Sweep root for stray files not cited by the guide; confirm `.venv`, PDFs, logs, PNGs all have a cited purpose (they do per §8).
5. `git -C SSD-offloading/LLMSimulator reset` (unstage the baseline — leaves the tree exactly as before Task 0, patch file preserved).
6. Check `git status` of both vendored repos shows no unexpected NEW untracked files (except `SSD-offloading/patch/offload_tier.patch`, which is intentional).

### Task 13: Final verification (all read-only, seconds)

```bash
./.venv/bin/python test_trace_io.py          # 8/8 passed
./.venv/bin/python analysis_d2.py --check   # all consistency checks passed
./.venv/bin/python analysis_d3.py            # runs on run_default.log
./.venv/bin/python analysis_d3.py run_synth64.log   # Mixtral example
./.venv/bin/python analysis_d4.py           # plots + ablation
```

Also verify one fresh podman container runs the SSD binary end-to-end (smoke).

All green → report summary to user: what changed, new numbers, where each claim now lives.
