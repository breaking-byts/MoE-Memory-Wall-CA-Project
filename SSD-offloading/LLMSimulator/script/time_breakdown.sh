#!/bin/bash

MAX_CORE=30


OUTPUT_DIR="../data/latency"

rm -rf run__.sh

mkdir -p ${OUTPUT_DIR}

for MODEL in mixtral
do
  for DATA in GSM ShareGPT Alpaca HellaSwag
  do
    for HARDWARE in "GPU" "LOGIC" "GPU+LOGIC"
    do
      for DEVICE_NUM in 2 4
      do
        for ITER in 100
        do
          echo "../build/timebreakdown-export ${MODEL} ${DATA} ${HARDWARE} ${DEVICE_NUM} ${OUTPUT_DIR} ${ITER}" >> run__.sh
        done
      done
    done
  done
done

./run_multiprog.py --process ${MAX_CORE} --script run__.sh
rm -rf run__.sh
