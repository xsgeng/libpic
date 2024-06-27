from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        name="libpic.current.cpu", 
        sources=["libpic/current/cpu.pyx"],
        include_dirs=["/home/chcl3/miniconda3/envs/libpic/lib/python3.12/site-packages/numpy/core/include"],
        # libraries=[...],
        # library_dirs=[...],
        extra_compile_args=['-march=native', '-fopenmp', '-O3', '-mavx2', '-mfma'],
        extra_link_args=['-fopenmp'],
    ),
]
setup(
    name="libpic",
    ext_modules=cythonize(extensions),
)