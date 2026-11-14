#!/usr/bin/env bash
# Final-evaluation pipeline (F0-F4).  Run from the repository root.
#   ./run_final.sh              build simulator + all experiments from the committed traces
#   ./run_final.sh --retrain    also regenerate the TinyMoE traces (fetch corpus, ~45 min training)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
SIM=SSD-offloading/LLMSimulator

echo "== 1. build the modified SSD-offloading simulator (incremental if already built)"
mkdir -p $SIM/build && cd $SIM/build
# -include cstdint: ramulator2 headers need it with g++ >= 13 (harmless with g++-11)
cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-include cstdint" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. > cmake.log
make -j"$(nproc)" run > make.log
cd "$ROOT"

echo "== 2. midterm D2 regression with the modified simulator (no routing trace)"
REG=experiments/sim_out/midterm_regression
mkdir -p $REG
rm -f $SIM/build/tmp/*.csv
for c in cpu cpu_copy; do
  cp ssd_cfgs/$c.yaml $SIM/build/config.yaml
  (cd $SIM/build && ./run > "$ROOT/$REG/ssd_$c.log")
done
cp $SIM/build/tmp/*.csv $REG/

echo "== 3. routing workloads"
python3 workloads/synthetic.py                       # uniform + Mixtral-calibrated (seconds)
if [ "${1:-}" = "--retrain" ]; then
  (cd workloads/tinymoe && ./fetch_corpus.sh && python3 train_tinymoe.py && python3 extract_traces.py)
fi

echo "== 4. unit tests"
python3 tests/test_prefetch.py

echo "== 5. F0 correlation analysis"
python3 experiments/f0_correlation.py
echo "== 6. F1 predictor accuracy"
python3 experiments/f1_predictor_accuracy.py
echo "== 7. F2-F4 prefetching vs on-demand (simulator + timeline)"
python3 experiments/f2_prefetch.py
echo "== 8. validation of simulator runs and cost model"
python3 experiments/validate_costs.py
echo "== 9. figures + headline table"
python3 experiments/f3_plots.py
echo "== 10. RESULTS.md"
python3 experiments/make_results_md.py
echo "done: results in experiments/results/ + RESULTS.md, figures in experiments/figures/"
