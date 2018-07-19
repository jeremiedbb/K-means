#!/bin/bash
n_iter=10

source deactivate

rm bench_sklearn_vs_intel.csv
rm points.csv
rm clusters.csv

echo 'step,distrib,pre-init,algo,tol,n_sample,n_feature,n_component,time,iter' >> bench_sklearn_vs_intel.csv

i=1
for s in 10000 100000
do
    for f in 100 1000 10000
    do
        for c in 10 100 1000 10000
        do
            if [ $c -ne $s ]
            then
                echo $i '/ 24' 
                source activate dev
                python make_points.py $s $f $c init >> bench_sklearn_vs_intel.csv

                # sklearn
                # pre init 'full' max_iter 
                python bench_sklearn_vs_intel.py $n_iter sklearn full 0 init >> bench_sklearn_vs_intel.csv
                # no pre init 'full' max_iter 
                python bench_sklearn_vs_intel.py $n_iter sklearn full 0 no-init >> bench_sklearn_vs_intel.csv
                # pre init 'elkan' max_iter 
                python bench_sklearn_vs_intel.py $n_iter sklearn elkan 0 init >> bench_sklearn_vs_intel.csv
                # no pre init 'elkan' max_iter 
                python bench_sklearn_vs_intel.py $n_iter sklearn elkan 0 no-init >> bench_sklearn_vs_intel.csv
                
                source deactivate

                # intel
                source activate intel
                # pre init 'full' max_iter 
                python bench_sklearn_vs_intel.py $n_iter intel full 0 init >> bench_sklearn_vs_intel.csv
                # no pre init 'full' max_iter 
                python bench_sklearn_vs_intel.py $n_iter intel full 0 no-init >> bench_sklearn_vs_intel.csv
                # pre init 'elkan' max_iter 
                python bench_sklearn_vs_intel.py $n_iter intel elkan 0 init >> bench_sklearn_vs_intel.csv
                # no pre init 'elkan' max_iter 
                python bench_sklearn_vs_intel.py $n_iter intel elkan 0 no-init >> bench_sklearn_vs_intel.csv
                source deactivate

                rm points.csv clusters.csv
                i=$(( i + 1 ))
            fi
        done
    done
done

