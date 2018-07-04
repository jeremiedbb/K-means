#!/bin/bash
n_iter=20

source /home/jeremie/intel/parallel_studio_xe_2018.3.051/psxevars.sh intel64
source /home/jeremie/intel/bin/compilervars.sh intel64
source deactivate

rm bench_kmeans_precompute.csv

export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_ENABLE_INSTRUCTIONS=AVX2

i=1
for s in 1000 10000 100000 1000000
do
    for f in 2 10 50 100
    do
        for c in 10 100 1000
        do
            echo $i '/ 48' 
            rm points.txt clusters.txt
            python make_points.py $s $f $c

            # sklearn
            python bench_kmeans_precompute.py $n_iter -sklearn 0 >> bench_kmeans_precompute.csv

            # intel
            source activate intel_python
            python bench_kmeans_precompute.py $n_iter -sklearn 1 >> bench_kmeans_precompute.csv
            source deactivate

            i=$(( i + 1 ))
        done
    done
done

