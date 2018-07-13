#!/bin/bash
n_iter=10

source deactivate

rm bench_sklearn_vs_intel.csv
rm points.csv
rm clusters.csv

i=1
for s in 10000 100000 1000000
do
    for f in 3 50
    do
        for c in 10 100 1000 10000
        do
            if [ $c -ne $s ]
            then
                echo $i '/ 24' 
                source activate dev
                python make_points.py $s $f $c init >> bench_sklearn_vs_intel.csv

                # sklearn
                python bench_sklearn_vs_intel.py $n_iter sklearn >> bench_sklearn_vs_intel.csv
                source deactivate

                # intel
                source activate intel
                python bench_sklearn_vs_intel.py $n_iter intel >> bench_sklearn_vs_intel.csv
                source deactivate

                rm points.csv clusters.csv
                i=$(( i + 1 ))
            fi
        done
    done
done

