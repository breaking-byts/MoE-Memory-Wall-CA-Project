#!/bin/bash

MAX_CORE=30


OUTPUT_DIR="../data/edap"

rm -rf run__.sh

mkdir -p ${OUTPUT_DIR}

echo "../build/edap ${OUTPUT_DIR}" >> run__.sh

./run_multiprog.py --process ${MAX_CORE} --script run__.sh
rm -rf run__.sh
