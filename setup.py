# setup.py

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION



from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "hough_transform_grayscale.pyx", compiler_directives={"language_level": "3"}, annotate=True
    ),
    include_dirs=[numpy.get_include()],
)

#python3 setup.py build_ext --inplace