from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import sys

if '-gcc' in sys.argv:
    compile_args = ['-ffast-math', '-fopenmp', '-fprefetch-loop-arrays']
    link_args = ["-fopenmp"]
    sys.argv.remove('-gcc')
    if '-avx2' in sys.argv:
        compile_args += ['-mavx2']
        sys.argv.remove('-avx2')
elif '-gcc8' in sys.argv:
    compile_args = ['-ffast-math', '-fopenmp', '-fprefetch-loop-arrays', '-falign-loops=32', '-fopt-info-vec', '-fopt-info-vec-missed']
    link_args = ["-fopenmp"]
    sys.argv.remove('-gcc8')
    if '-avx2' in sys.argv:
        compile_args += ['-mavx2']
        sys.argv.remove('-avx2')
elif '-icc' in sys.argv:
    compile_args = ['-qopenmp', '-fprefetch-array-loops', '-qopt-report-phase=vec', '-qopt-report=2']
    link_args = ["-qopenmp"]
    sys.argv.remove('-icc')
    if '-avx2' in sys.argv:
        compile_args += ['-march=core_avx2']
        sys.argv.remove('-avx2')
else:
    compile_args = []
    link_args = []

extensions = [Extension("looping",
                        sources=["looping.pyx"],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[numpy.get_include()])]

setup(
    ext_modules = cythonize(extensions, annotate=True)
)