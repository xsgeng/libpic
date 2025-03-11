from setuptools import Extension, setup
# from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="libpic.current.cpu",
        sources=["libpic/current/cpu.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.interpolation.cpu",
        sources=["libpic/interpolation/cpu.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
    ),
    Extension(
        name="libpic.sort.cpu", 
        sources=["libpic/sort/cpu.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.pusher.unified.cpu", 
        sources=["libpic/pusher/unified/cpu.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.patch.sync_fields", 
        sources=["libpic/patch/sync_fields.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.patch.sync_particles", 
        sources=["libpic/patch/sync_particles.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    )
]
setup(
    name="libpic",
    ext_modules=extensions,
)