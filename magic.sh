n_sample=100000
n_dim=128
n_cluster=1024

n_iter_sklearn=1
n_iter_ff=1
n_iter_j=1

source /home/jeremie/intel/parallel_studio_xe_2018.3.051/psxevars.sh intel64
source /home/jeremie/intel/bin/compilervars.sh intel64
source deactivate

rm results log

# sklearn vanilla
echo "***** sklearn vanilla *****" >> results
echo "" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc build_ext --inplace >> log
python run.py $n_sample $n_dim $n_cluster $n_iter_sklearn 1 >> results
echo "" >> results

# sklearn intel
echo "****** sklearn intel ******" >> results
echo "" >> results
source activate intel_python
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc build_ext --inplace >> log
python run.py $n_sample $n_dim $n_cluster $n_iter_sklearn 1 >> results
source deactivate
echo "" >> results

# buil gcc (no avx2 ?)
echo "******* GCC no AVX2 *******" >> results
echo "" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc build_ext --inplace >> log
python run.py $n_sample $n_dim $n_cluster $n_iter_ff 2 >> results
python run.py $n_sample $n_dim $n_cluster $n_iter_j 4 >> results
echo "" >> results
 
# build gcc avx2
echo "******** GCC AVX2 *********" >> results
echo "" >> results
source deactivate
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc -avx2 build_ext --inplace >> log
python run.py $n_sample $n_dim $n_cluster $n_iter_ff 2 >> results
python run.py $n_sample $n_dim $n_cluster $n_iter_j 4 >> results
echo "" >> results

# build icc (no avx2 ?)
echo "****** ICC no AVX2 ********" >> results
echo "" >> results
source deactivate
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
LDSHARED="icc -shared" CC=icc python setup.py -icc build_ext --inplace >> log
python run.py $n_sample $n_dim $n_cluster $n_iter_ff 2 >> results
python run.py $n_sample $n_dim $n_cluster $n_iter_j 4 >> results
echo "" >> results

# build gcc avx2
echo "******** ICC AVX2 *********" >> results
echo "" >> results
source deactivate
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
LDSHARED="icc -shared" CC=icc python setup.py -icc -avx2 build_ext --inplace >> log
python run.py $n_sample $n_dim $n_cluster $n_iter_ff 2 >> results
python run.py $n_sample $n_dim $n_cluster $n_iter_j 4 >> results
echo "" >> results