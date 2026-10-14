#!/bin/bash

MAX_CORE=32


OUTPUT_DIR="../data/sum_gen_split"

rm -rf run__.sh

mkdir -p ${OUTPUT_DIR}

for MODEL in "mixtral"
do
  for DATA in "synthesis"
  do
    for HARDWARE in "GPU" 
    do
      for DEVICE_NUM in 2
      do
        for MAX_BATCHSIZE in 64
        do
          for MAX_PROCESS_TOKEN in 0
          do
            for INPUT_LEN in 1024
            do
              for OUTPUT_LEN in 16 64 256 1024 4096
              do
                for ITER in 5000
                do
                  for INJECTION_RATE in 0 1 2 4 8 16
                  do
                    for NUM_NODE in 1
                    do
                      for PARALLEL_EXECUTION in 0
                      do
                          echo "../build/sum_gen_split ${MODEL} ${DATA} ${HARDWARE} ${DEVICE_NUM} ${OUTPUT_DIR} ${MAX_BATCHSIZE} ${INPUT_LEN} ${OUTPUT_LEN} ${MAX_PROCESS_TOKEN} ${INJECTION_RATE} ${ITER} 1 1 ${PARALLEL_EXECUTION}" >> run__.sh
                  
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
  done
done

./run_multiprog.py --process ${MAX_CORE} --script run__.sh
rm -rf run__.sh
