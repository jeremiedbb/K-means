from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

#
compile_args = []
link_args = []

extensions = [Extension("kmeans_chunked",
                        sources=["kmeans_chunked.pyx"],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[numpy.get_include()])]

setup(
    ext_modules=cythonize(extensions, annotate=True, force=True)
)
