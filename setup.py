from setuptools import Extension, setup
# from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="libpic.current.cpu",
        sources=["libpic/current/cpu.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
    ),
    Extension(
        name="libpic.sort.cpu", 
        sources=["libpic/sort/cpu.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
    ),
    Extension(
        name="libpic.utils.clist", 
        sources=["libpic/utils/clist.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
    ),
]
setup(
    name="libpic",
    ext_modules=extensions,
)