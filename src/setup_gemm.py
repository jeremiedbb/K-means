from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import sys

extensions = [Extension("gemm",
                        sources=["test_gemm.pyx"],
                        extra_compile_args=['-ffast-math', '-fopenmp'],
                        extra_link_args=["-fopenmp"],
                        include_dirs=[numpy.get_include()])]

setup(
    ext_modules = cythonize(extensions, annotate=True)
)