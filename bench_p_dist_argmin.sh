n_sample=100000
n_feature=3
n_component=1024
n_iter=5

rm results log
rm points.csv

python make_points.py $n_sample $n_feature

# sklearn pairwise dist argmin
echo "---------- sklearn pairwise dist argmin -------------" >> results
rm -rf build/
rm square_dist.html square_dist.c*
python setup_sqd.py build_ext --inplace >> log
for i in 0.1 0.25 0.5 0.75 1 2 3 4 5 6 7 8 9 10 25 50 100 250 500 750 1000
do
    echo " Working memory "$i"M" >> results
    python run_sqd.py $n_component $n_iter 3 $i >> results
    echo "" >> results
done
echo "-----------------------------------------------------" >> results
echo "" >> results
echo "" >> results

