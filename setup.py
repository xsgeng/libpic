from setuptools import Extension, setup
# from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="libpic.current.cpu2d",
        sources=["libpic/current/cpu2d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.current.cpu3d",
        sources=["libpic/current/cpu3d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.interpolation.cpu2d",
        sources=["libpic/interpolation/cpu2d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.interpolation.cpu3d",
        sources=["libpic/interpolation/cpu3d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.sort.cpu2d", 
        sources=["libpic/sort/cpu2d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.sort.cpu3d", 
        sources=["libpic/sort/cpu3d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.pusher.unified.unified_pusher_2d", 
        sources=["libpic/pusher/unified/unified_pusher_2d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.pusher.unified.unified_pusher_3d", 
        sources=["libpic/pusher/unified/unified_pusher_3d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.patch.sync_fields2d", 
        sources=["libpic/patch/sync_fields2d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.patch.sync_fields3d", 
        sources=["libpic/patch/sync_fields3d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.patch.sync_particles_2d", 
        sources=["libpic/patch/sync_particles_2d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        name="libpic.patch.sync_particles_3d", 
        sources=["libpic/patch/sync_particles_3d.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize'],
        extra_link_args=['-fopenmp'],
    ),
]
setup(
    name="libpic",
    ext_modules=extensions,
)