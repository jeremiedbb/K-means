#!/bin/bash
n_iter=10

source deactivate

export MKL_NUM_THREADS=32
export MKL_ENABLE_INSTRUCTIONS=AVX

rm bench_kmeans_wm.csv

export PYTHONPATH=${HOME}/scikit-learn:${PYTHONPATH}

python bench_kmeans_wm.py