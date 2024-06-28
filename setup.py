from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="libpic.current.cpu", 
        sources=["libpic/current/cpu.pyx"],
        include_dirs=[np.get_include()],
        # libraries=[...],
        # library_dirs=[...],
        extra_compile_args=['-march=native', '-fopenmp', '-O3', '-mavx2', '-mfma'],
        extra_link_args=['-fopenmp'],
    ),
]
setup(
    name="libpic",
    ext_modules=extensions,
)