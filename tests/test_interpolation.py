import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.interpolation.cpu import _interpolation_2d, interpolation_patches_2d
from libpic.interpolation.cpu3d import interpolation_patches_3d
from libpic.fields import Fields3D, Fields2D
from libpic.particles import ParticlesBase

class TestCurrentDeposition(unittest.TestCase):

    def test_interp2d(self):
        """test if interpolation_2d works"""
        npart = 3
        x = np.array([-2.0, np.nan, 1.0])
        y = np.full_like(x, 1.0)
        ex_part = np.full_like(x, 0)
        ey_part = np.full_like(x, 0)
        ez_part = np.full_like(x, 0)
        bx_part = np.full_like(x, 0)
        by_part = np.full_like(x, 0)
        bz_part = np.full_like(x, 0)
        is_dead = np.full(npart, False)
        is_dead[1] = True
        x[1] = np.nan

        nx = 3
        ny = 3
        x0 = 0.0
        y0 = 0.0
        dx = 1.0
        dy = 1.0
        
        ex = np.arange(nx*ny, dtype='f8').reshape((nx, ny))
        ey = ex
        ez = np.zeros((nx, ny))
        bx = np.zeros((nx, ny))
        by = np.zeros((nx, ny))
        bz = np.zeros((nx, ny))

        _interpolation_2d(
            x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, 
            is_dead,
            npart,
            ex, ey, ez, bx, by, bz,
            dx, dy, x0, y0, nx, ny,
        )
        self.assertSequenceEqual(ex_part.tolist(), [2.5, 0, 2.5])
        self.assertSequenceEqual(ey_part.tolist(), [3.5, 0, 3.5])

    def test_list(self):
        npatch = 128
        nx = 100
        ny = 100
        npart = 100000

        dx = 1e-6
        dy = 1e-6
        dt = dx / 2 / c
        q = e

        # Example usage
        ex_list = [np.arange(nx*ny, dtype=float).reshape((nx, ny))*1e15 for _ in range(npatch)]
        ey_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        ez_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        bx_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        by_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        bz_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        x0_list = [0.0 for _ in range(npatch)]
        y0_list = [0.0 for _ in range(npatch)]

        x_list = [np.random.uniform(10*dx, (nx-10)*dx, npart) for _ in range(npatch)]
        y_list = [np.random.uniform(10*dy, (ny-10)*dy, npart) for _ in range(npatch)]
        ex_part_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        ey_part_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        ez_part_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        bx_part_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        by_part_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        bz_part_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        is_dead_list = [np.full(npart, False) for _ in range(npatch)]

        interpolation_patches_2d(
            x_list, y_list,
            ex_part_list, ey_part_list, ez_part_list,
            bx_part_list, by_part_list, bz_part_list,
            is_dead_list,
            ex_list, ey_list, ez_list,
            bx_list, by_list, bz_list,
            x0_list, y0_list,
            npatch,
            dx, dy,
            nx, ny,
        ) 
        print(ex_part_list[0])

        tic = perf_counter_ns()

        toc = perf_counter_ns()

        # import os
        # nthreads = int(os.getenv('OMP_NUM_THREADS', os.cpu_count()))
        # nthreads = min(nthreads, npatch)
        # print(f"current_deposit_2d {(toc - tic)/(npart*npatch)*nthreads:.0f} ns per particle")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1)

        h = axes[0].imshow(ex_list[0].T)
        fig.colorbar(h, ax=axes[0])
        h = axes[1].scatter(x_list[0], y_list[0], c=ex_part_list[0], s=2)
        fig.colorbar(h, ax=axes[1])

        fig.savefig("test.png", dpi=300)

    def test_interp3d(self):
        nx = 3
        ny = 3
        nz = 3
        dx = dy = dz = 1.0
        npart = 3

        particles = ParticlesBase(0, 0)
        particles.attrs += ['z']
        particles.initialize(npart)

        fields = Fields3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, x0=0, y0=0, z0=0, n_guard=3)

        particles.x[:] = np.array([-2.0, np.nan, 1.0])
        particles.y[:] = 1
        particles.z[:] = 1
        particles.is_dead[1] = True

        fields.ex[:nx, :ny, :nz] = np.arange(nx*ny, dtype='f8').reshape((nx, ny))[:, :, None]
        fields.ey[:nx, :ny, :nz] = fields.ex[:nx, :ny, :nz]

        interpolation_patches_3d(
            [particles], [fields], 1
        )
        print(particles.ex_part)
        self.assertSequenceEqual(particles.ex_part.tolist(), [0.0, 0, 2.5])
        self.assertSequenceEqual(particles.ey_part.tolist(), [0.0, 0, 3.5])