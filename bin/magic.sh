n_sample=500000
n_dim=32
n_cluster=32

n_iter_sklearn=5
n_iter_ff=5
n_iter_j=5

source /home/jeremie/intel/parallel_studio_xe_2018.3.051/psxevars.sh intel64
source /home/jeremie/intel/bin/compilervars.sh intel64
source deactivate

rm results log
rm points.csv

python make_points.py $n_sample $n_dim

# sklearn vanilla
echo "---------- sklearn vanilla ----------------------------------------------" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc build_ext --inplace >> log
python run.py $n_cluster $n_iter_sklearn 1 >> results
echo "-------------------------------------------------------------------------" >> results
echo "" >> results
echo "" >> results

# sklearn intel
echo "---------- sklearn intel ------------------------------------------------" >> results
source activate intel_python
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc build_ext --inplace >> log
python run.py $n_cluster $n_iter_sklearn 1 >> results
source deactivate
echo "-------------------------------------------------------------------------" >> results
echo "" >> results
echo "" >> results

# buil gcc (no avx2 ?)
echo "---------- gcc ----------------------------------------------------------" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc build_ext --inplace >> log
python run.py $n_cluster $n_iter_ff 2 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 4 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 5 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 6 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 7 >> results
echo "-------------------------------------------------------------------------" >> results
echo "" >> results
echo "" >> results
 
# build gcc avx2
echo "---------- gcc AVX2 -----------------------------------------------------" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
python setup.py -gcc -avx2 build_ext --inplace >> log
python run.py $n_cluster $n_iter_ff 2 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 4 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 5 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 6 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 7 >> results
echo "-------------------------------------------------------------------------" >> results
echo "" >> results
echo "" >> results

# build icc (no avx2 ?)
echo "---------- icc ----------------------------------------------------------" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
LDSHARED="icc -shared" CC=icc python setup.py -icc build_ext --inplace >> log
python run.py $n_cluster $n_iter_ff 2 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 4 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 5 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 6 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 7 >> results
echo "-------------------------------------------------------------------------" >> results
echo "" >> results
echo "" >> results

# build gcc avx2
echo "---------- icc AVX2 -----------------------------------------------------" >> results
rm -rf build/
rm misc_kmeans.html misc_kmeans.c*
LDSHARED="icc -shared" CC=icc python setup.py -icc -avx2 build_ext --inplace >> log
python run.py $n_cluster $n_iter_ff 2 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 4 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 5 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 6 >> results
echo "" >> results
python run.py $n_cluster $n_iter_j 7 >> results
echo "-------------------------------------------------------------------------" >> results
echo "" >> results
echo "" >> results