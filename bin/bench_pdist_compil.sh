#!/bin/bash
n_sample=100000
n_feature=100
n_component=1000
n_iter=20

    gcc5noAVX2=1
      gcc5AVX2=1
    gcc8noAVX2=1
      gcc8AVX2=1
     iccnoAVX2=1
       iccAVX2=1

source /home/jeremie/intel/parallel_studio_xe_2018.3.051/psxevars.sh intel64
source /home/jeremie/intel/bin/compilervars.sh intel64
source deactivate

export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_ENABLE_INSTRUCTIONS=AVX

rm results log
rm points.csv

python make_points.py $n_sample $n_feature

function clean {
    rm -rf build/
    rm square_dist.html square_dist.c*
}

function run {
    for i in 5
    do
        python run_sqd.py $n_component $n_iter $i >> bench_pdist_compil_100.csv
    done
}

function end {
    echo "-----------------------------------------------------" >> results
    echo "" >> results
    echo "" >> results
}

# gcc-5 no AVX2
if [ $gcc5noAVX2 -eq 1 ]
then
    export MKL_ENABLE_INSTRUCTIONS=AVX
    echo "---------- gcc-5 ------------------------------------" >> results
    echo "" >> results
    clean
    python setup_sqd.py -gcc build_ext --inplace >> log
    run
    end
fi

# gcc-5 AVX2
if [ $gcc5AVX2 -eq 1 ]
then
    export MKL_ENABLE_INSTRUCTIONS=AVX2
    echo "---------- gcc-5 AVX2 -------------------------------" >> results
    echo "" >> results
    clean
    python setup_sqd.py -gcc -avx2 build_ext --inplace >> log
    run
    end
fi

# gcc-8 no AVX2
if [ $gcc8noAVX2 -eq 1 ]
then
    export MKL_ENABLE_INSTRUCTIONS=AVX
    echo "---------- gcc-8 ------------------------------------" >> results
    echo "" >> results
    clean
    python setup_sqd.py -gcc8 build_ext --inplace >> log
    run
    end
fi

# gcc-8 AVX2
if [ $gcc8AVX2 -eq 1 ]
then
    export MKL_ENABLE_INSTRUCTIONS=AVX2
    echo "---------- gcc-8 AVX2 -------------------------------" >> results
    echo "" >> results
    clean
    python setup_sqd.py -gcc8 -avx2 build_ext --inplace >> log
    run
    end
fi

# icc-8 no AVX2
if [ $iccnoAVX2 -eq 1 ]
then
    export MKL_ENABLE_INSTRUCTIONS=AVX
    echo "---------- icc --------------------------------------" >> results
    echo "" >> results
    clean
    LDSHARED="icc -shared" CC=icc python setup_sqd.py -icc build_ext --inplace >> log
    run
    end
fi

# icc-8 AVX2
if [ $iccAVX2 -eq 1 ]
then
    export MKL_ENABLE_INSTRUCTIONS=AVX2
    echo "---------- icc AVX2 ---------------------------------" >> results
    echo "" >> results
    clean
    LDSHARED="icc -shared" CC=icc python setup_sqd.py -icc -avx2 build_ext --inplace >> log
    run
    end
fi