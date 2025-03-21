import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.interpolation.cpu2d import interpolation_patches_2d
from libpic.interpolation.cpu3d import interpolation_patches_3d
from libpic.fields import Fields3D, Fields2D
from libpic.particles import ParticlesBase

class TestCurrentDeposition(unittest.TestCase):
    def test_interp2d(self):
        nx = 3
        ny = 3
        dx = dy = 1.0
        npart = 3

        particles = ParticlesBase(0, 0)
        particles.initialize(npart)

        fields = Fields2D(nx=nx, ny=ny, dx=dx, dy=dy, x0=0, y0=0, n_guard=3)

        particles.x[:] = np.array([-2.0, np.nan, 1.0])
        particles.y[:] = 1
        particles.z[:] = 1
        particles.is_dead[1] = True

        fields.ex[:nx, :ny] = np.arange(nx*ny, dtype='f8').reshape((nx, ny))
        fields.ey[:nx, :ny] = fields.ex[:nx, :ny]

        interpolation_patches_2d(
            [particles], [fields], 1
        )
        self.assertSequenceEqual(particles.ex_part.tolist(), [0.0, 0, 2.5])
        self.assertSequenceEqual(particles.ey_part.tolist(), [0.0, 0, 3.5])

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