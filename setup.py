#from distutils.core import setup
import numpy

from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'


ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    ),
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]

setup(
    name='parallel-tutorial',
    ext_modules=cythonize(
        "hough_transform_grayscale.pyx", 
        compiler_directives={"language_level": "3"}, 
        annotate=True
    ),
    include_dirs=[numpy.get_include()],
)

#python3 setup.py build_ext --inplace
