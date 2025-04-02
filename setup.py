import os
import subprocess
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

include_dirs = [np.get_include()]
extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize']
extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        name="libpic.current.cpu2d",
        sources=["libpic/current/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.current.cpu3d",
        sources=["libpic/current/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.interpolation.cpu2d",
        sources=["libpic/interpolation/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.interpolation.cpu3d",
        sources=["libpic/interpolation/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.sort.cpu2d", 
        sources=["libpic/sort/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.sort.cpu3d", 
        sources=["libpic/sort/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.pusher.unified.unified_pusher_2d", 
        sources=["libpic/pusher/unified/unified_pusher_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.pusher.unified.unified_pusher_3d", 
        sources=["libpic/pusher/unified/unified_pusher_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.patch.sync_fields2d", 
        sources=["libpic/patch/sync_fields2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.patch.sync_fields3d", 
        sources=["libpic/patch/sync_fields3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.patch.sync_particles_2d", 
        sources=["libpic/patch/sync_particles_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="libpic.patch.sync_particles_3d", 
        sources=["libpic/patch/sync_particles_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

class QEDBuildCommand(build_ext):
    """集成HDF5表生成的官方推荐实现"""
    def run(self):
        # 标准构建流程
        super().run()

        # 确保在构建环境可用后才生成
        self._generate_qed_tables()

    def _generate_qed_tables(self):
        """使用独立进程生成表数据"""

        for table in ["optical_depth_tables_sigmoid", "optical_depth_tables"]:
            if os.path.exists(f"libpic/qed/{table}.h5"):
                continue
            gen_script = os.path.join(
                os.path.dirname(__file__),
                f"libpic/qed/{table}.py"
            )
      
            # 在独立Python进程中执行生成
            cmd = [
                sys.executable,
                gen_script,
            ]

            print(f"\n🚀 Generating {table}:")
            print(" ".join(cmd))
            subprocess.check_call(cmd)

setup(
    name="libpic",
    ext_modules=extensions,
    cmdclass={
        'build_ext': QEDBuildCommand
    },
)