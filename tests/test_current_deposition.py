import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.current.cpu import current_deposition_cpu


class TestCurrentDeposition(unittest.TestCase):
    def test_precision(self):
        nx = 6
        ny = 6
        npart = 1
        dx = dy = 1.0e-6
        x0_list = [-3*dx]
        y0_list = [-3*dy]
        dt = dx / c * 0.9
        q = e

        ne = 1e27
        w_list = [np.array([ne*dx*dy])]
        
        ux_list = [np.random.uniform(-10.0, 10.0, (1,))]
        uy_list = [np.random.uniform(-10.0, 10.0, (1,))]
        uz_list = [np.random.uniform(-10.0, 10.0, (1,))]
        inv_gamma_list = [1 / np.sqrt(1 + ux_list[0]**2 + uy_list[0]**2 + uz_list[0]**2)]

        rho_list = [np.zeros((nx, ny))]
        jx_list = [np.zeros((nx, ny))]
        jy_list = [np.zeros((nx, ny))]
        jz_list = [np.zeros((nx, ny))]

        is_dead_list = [np.full(npart, False)]

        x_list = [np.random.uniform(-dx, dx, (1,))]
        y_list = [np.random.uniform(-dy, dy, (1,))]
        current_deposition_cpu(
            rho_list, jx_list, jy_list, jz_list, 
            x0_list, y0_list, 
            x_list, y_list, 
            ux_list, uy_list, uz_list, 
            inv_gamma_list, 
            is_dead_list, 
            w_list,
            1, dx, dy, dt, q
        )
        vx = ux_list[0]*inv_gamma_list[0]*c
        vy = uy_list[0]*inv_gamma_list[0]*c
        vz = uz_list[0]*inv_gamma_list[0]*c

        self.assertLess(abs(jx_list[0].sum() - q*ne*vx)/(q*ne*vx), 1e-10)
        self.assertLess(abs(jy_list[0].sum() - q*ne*vy)/(q*ne*vy), 1e-10)
        self.assertLess(abs(jz_list[0].sum() - q*ne*vz)/(q*ne*vz), 1e-10)
        self.assertLess(abs(rho_list[0].sum() - ne*q)/(ne*q), 1e-10)

    def test_numba_func(self):
        npatch = 128
        nx = 100
        ny = 100
        npart = 100000

        dx = 1e-6
        dy = 1e-6
        dt = dx / 2 / c
        q = e

        # Example usage
        rho_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        jx_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        jy_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        jz_list = [np.zeros((nx, ny)) for _ in range(npatch)]
        x0_list = [0.0 for _ in range(npatch)]
        y0_list = [0.0 for _ in range(npatch)]
        x_list = [np.random.uniform(10*dx, (nx-10)*dx, npart) for _ in range(npatch)]
        y_list = [np.random.uniform(10*dy, (ny-10)*dy, npart) for _ in range(npatch)]
        ux_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        uy_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        uz_list = [np.random.uniform(-0, 0, npart) for _ in range(npatch)]
        inv_gamma_list = [1/np.sqrt(ux**2 + uy**2 + uz**2 + 1) for ux, uy, uz in zip(ux_list, uy_list, uz_list)]
        is_dead_list = [np.full(npart, False) for _ in range(npatch)]
        w_list = [np.random.rand(npart) for _ in range(npatch)]

        current_deposition_cpu(
            rho_list, jx_list, jy_list, jz_list, 
            x0_list, y0_list, 
            x_list, y_list, 
            ux_list, uy_list, uz_list, 
            inv_gamma_list, 
            is_dead_list, 
            w_list,
            npatch, dx, dy, dt, q
        )    
        self.assertFalse(np.isnan(rho_list[0]).any())
        self.assertFalse(np.isnan(jx_list[0]).any())
        self.assertFalse(np.isnan(jy_list[0]).any())
        self.assertFalse(np.isnan(jz_list[0]).any())

        tic = perf_counter_ns()
        current_deposition_cpu(
            rho_list, jx_list, jy_list, jz_list, 
            x0_list, y0_list, 
            x_list, y_list, 
            ux_list, uy_list, uz_list, 
            inv_gamma_list, 
            is_dead_list, 
            w_list,
            npatch, dx, dy, dt, q
        )
        toc = perf_counter_ns()

        import os
        nthreads = int(os.getenv('OMP_NUM_THREADS', os.cpu_count()))
        nthreads = min(nthreads, npatch)
        print(f"current_deposit_2d {(toc - tic)/(npart*npatch)*nthreads:.0f} ns per particle")

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