#!/bin/bash
n_iter=10

source deactivate

export MKL_NUM_THREADS=32
export MKL_ENABLE_INSTRUCTIONS=AVX

rm bench_sklearn_vs_intel.csv

i=1
for s in 10000 100000 1000000
do
    for f in 3 50
    do
        for c in 10 1000 10000
        do
            if [ $c -ne $s ]
            then
                echo $i '/ 16' 
                export PYTHONPATH=${HOME}/scikit-learn:${PYTHONPATH}
                python make_points.py $s $f $c >> bench_sklearn_vs_intel.csv

                # sklearn
                python bench_sklearn_vs_intel.py $n_iter sklearn >> bench_sklearn_vs_intel.csv

                # intel
                unset PYTHONPATH
                source activate intel_python
                python bench_sklearn_vs_intel.py $n_iter intel >> bench_sklearn_vs_intel.csv
                source deactivate

                rm points.csv clusters.csv
                i=$(( i + 1 ))
            fi
        done
    done
done

