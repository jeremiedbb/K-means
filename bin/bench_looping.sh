python setup_looping.py build_ext --inplace

python bench_looping.py

rm -rf build/
rm *.c
rm *.html
rm *.so
