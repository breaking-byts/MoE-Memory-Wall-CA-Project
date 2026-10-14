#!/bin/bash

MAX_CORE=30


OUTPUT_DIR="../data/latency"
# OUTPUT_DIR="../data/latency/not_ramul"

rm -rf run__.sh

mkdir -p ${OUTPUT_DIR}

for MODEL in mixtral
do
  for DATA in "synthesis"
  do
    for HARDWARE in "GPU"
    do
      for DEVICE_NUM in 4
      do
        for MAX_BATCHSIZE in 64
        do
          for MAX_PROCESS_TOKEN in 0
          do
            for INPUT_LEN in 4096
            do
              for OUTPUT_LEN in 512
              do
                for INJECTION_RATE in 0
                do
                  for ITER in 1000
                  do
                    echo "../build/stat-export ${MODEL} ${DATA} ${HARDWARE} ${DEVICE_NUM} ${OUTPUT_DIR} ${MAX_BATCHSIZE} ${INPUT_LEN} ${OUTPUT_LEN} ${MAX_PROCESS_TOKEN} ${INJECTION_RATE} ${ITER}" >> run__.sh
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

./run_multiprog.py --process ${MAX_CORE} --script run__.sh
rm -rf run__.sh
