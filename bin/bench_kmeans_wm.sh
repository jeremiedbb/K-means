#!/bin/bash
source deactivate

export MKL_NUM_THREADS=4
export MKL_ENABLE_INSTRUCTIONS=AVX2

rm ../bench/bench_kmeans_wm.csv

export PYTHONPATH=${HOME}/scikit-learn:${PYTHONPATH}

python ../src/bench_kmeans_wm.py
