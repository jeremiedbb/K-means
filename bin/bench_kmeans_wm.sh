#!/bin/bash
source deactivate

rm ../bench/bench_kmeans_wm.csv

export PYTHONPATH=${HOME}/scikit-learn:${PYTHONPATH}

python ../src/bench_kmeans_wm.py
