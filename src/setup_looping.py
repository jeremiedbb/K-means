from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


compile_args = ['-ffast-math',
                '-fopenmp',
                '-fopt-info-vec']
link_args = ["-fopenmp"]

extensions = [Extension("looping",
                        sources=["looping.pyx"],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[numpy.get_include()])]

setup(
    ext_modules=cythonize(extensions, annotate=True)
)
