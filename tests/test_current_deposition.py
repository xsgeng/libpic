import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e
from libpic.current.cpu import current_deposit_2d

from libpic.current.deposition import CurrentDeposition2D


class TestCurrentDeposition(unittest.TestCase):
    def test_precision(self):
        nx = 6
        ny = 6
        npart = 1
        dx = dy = 1.0e-6
        x0 = -3*dx
        y0 = -3*dy
        dt = dx / c * 0.9
        q = e

        ne = 1e27
        w = np.array([ne*dx*dy])
        
        ux = np.random.uniform(-10.0, 10.0, (1,))
        uy = np.random.uniform(-10.0, 10.0, (1,))
        uz = np.random.uniform(-10.0, 10.0, (1,))
        inv_gamma = 1 / np.sqrt(1 + ux**2 + uy**2 + uz**2)

        rho = np.zeros((nx, ny))
        jx = np.zeros((nx, ny))
        jy = np.zeros((nx, ny))
        jz = np.zeros((nx, ny))

        pruned = np.full(npart, False)

        x = np.random.uniform(-dx, dx, (1,))
        y = np.random.uniform(-dy, dy, (1,))
        current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)
        vx = ux*inv_gamma*c
        vy = uy*inv_gamma*c
        vz = uz*inv_gamma*c

        self.assertLess(abs(jx.sum() - q*ne*vx)/(q*ne*vx), 1e-10)
        self.assertLess(abs(jy.sum() - q*ne*vy)/(q*ne*vy), 1e-10)
        self.assertLess(abs(jz.sum() - q*ne*vz)/(q*ne*vz), 1e-10)
        self.assertLess(abs(rho.sum() - ne*q)/(ne*q), 1e-10)

    def test_numba_func(self):
        nx = 100
        ny = 100
        npart = 1000000
        x0 = 0.0
        y0 = 0.0
        dx = 1.0e-6
        dy = 1.0e-6
        lx = nx * dx
        ly = ny * dy
        dt = dx / c / 2
        q = e

        w = np.ones(npart)
        x = np.random.uniform(low=3*dx, high=lx-3*dx, size=npart)
        y = np.random.uniform(low=3*dy, high=ly-3*dy, size=npart)
        ux = np.random.uniform(low=-10.0, high=10.0, size=npart)
        uy = np.random.uniform(low=-10.0, high=10.0, size=npart)
        uz = np.random.uniform(low=-10.0, high=10.0, size=npart)
        inv_gamma = 1 / np.sqrt(1 + ux**2 + uy**2 + uz**2)

        idx = np.argsort(x/dx + nx*y/dy)
        x[:] = x[idx]
        y[:] = y[idx]
        ux[:] = ux[idx]
        uy[:] = uy[idx]
        uz[:] = uz[idx]
        inv_gamma[:] = inv_gamma[idx]

        # Jrho = np.zeros((4, nx, ny), order="C")
        # jx = Jrho[0, :, :]
        # jy = Jrho[1, :, :]
        # jz = Jrho[2, :, :]
        # rho = Jrho[3, :, :]
        rho = np.zeros((nx, ny))
        jx = np.zeros((nx, ny))
        jy = np.zeros((nx, ny))
        jz = np.zeros((nx, ny))

        pruned = np.full(npart, False)

        current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)
        self.assertFalse(np.isnan(rho).any())
        self.assertFalse(np.isnan(jx).any())
        self.assertFalse(np.isnan(jy).any())
        self.assertFalse(np.isnan(jz).any())

        tic = perf_counter_ns()
        current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)
        toc = perf_counter_ns()
        print(f"current_deposit_2d {(toc - tic)/1e6:.0f} ms")
        print(f"current_deposit_2d {(toc - tic)/npart:.0f} ns per particle")

    # def test_current_deposition_class(self):
    #     from libpic.fields import Fields2D
    #     from libpic.particles import ParticlesBase
    #     from libpic.species import Electron
    #     from libpic.patch import Patch2D, Patches

    #     nx = 64
    #     ny = 64
    #     npart = 100
    #     x0 = 0.0
    #     y0 = 0.0
    #     dx = 1.0e-6
    #     dy = 1.0e-6
    #     lx = nx * dx
    #     ly = ny * dy
    #     dt = dx / c / 2
    #     q = e

        
    #     field = Fields2D(nx, ny, dx, dy, x0, y0, n_guard=3)
    #     patch = Patch2D(0, 0, 0, 0, x0, y0, field)
    #     patches = Patches()
    #     patches.append(patch)

    #     current_deposition = CurrentDeposition2D(patches)
        
    #     species = Electron(density=lambda x, y: 1.0)
    #     particles = ParticlesBase(species=species)
    #     patches.add_species(species)

    #     particles.initialize(npart=npart)
    #     particles.w[:]  = 1
    #     particles.x[:]  = np.random.uniform(low=3*dx, high=lx-3*dx, size=npart)
    #     particles.y[:]  = np.random.uniform(low=3*dy, high=ly-3*dy, size=npart)
    #     particles.ux[:] = np.random.uniform(low=-1.0, high=1.0, size=npart)
    #     particles.uy[:] = np.random.uniform(low=-1.0, high=1.0, size=npart)
    #     particles.uz[:] = np.random.uniform(low=-1.0, high=1.0, size=npart)
    #     particles.inv_gamma[:] = 1 / np.sqrt(1 + particles.ux**2 + particles.uy**2 + particles.uz**2)

    #     current_deposition.update_patches()

    #     current_deposition(ispec=0, dt=dt)