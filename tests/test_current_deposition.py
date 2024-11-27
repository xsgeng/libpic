import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.current.cpu import current_deposition_cpu
from libpic.current.deposition import CurrentDeposition2D


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

    def test_current_deposition_class(self):
        from scipy.constants import c, pi
        from libpic.fields import Fields2D
        from libpic.patch.patch import Patch2D, Patches
        from libpic.species import Electron, Proton
        
        l0 = 0.8e-6
        nc = 1.74e27

        dx = 1e-6
        dy = 1e-6

        nx = 128
        ny = 128

        npatch_x = 4
        npatch_y = 4

        nx_per_patch = nx//npatch_x
        ny_per_patch = ny//npatch_y

        Lx = nx*dx
        Ly = ny*dy

        n_guard = 3
        patches = Patches(dimension=2)
        for j in range(npatch_y):
            for i in range(npatch_x):
                index = i + j * npatch_x
                p = Patch2D(
                    rank=0, 
                    index=index, 
                    ipatch_x=i, 
                    ipatch_y=j, 
                    x0=i*Lx/npatch_x, 
                    y0=j*Ly/npatch_y,
                    nx=nx_per_patch, 
                    ny=ny_per_patch, 
                    dx=dx,
                    dy=dy,
                )
                f = Fields2D(nx=nx_per_patch, ny=ny_per_patch, dx=dx,dy=dy, x0=i*Lx/npatch_x, y0=j*Ly/npatch_y, n_guard=n_guard)      
                p.set_fields(f)

                if i > 0:
                    p.set_neighbor_index(xmin=(i - 1) + j * npatch_x)
                if i < npatch_x - 1:
                    p.set_neighbor_index(xmax=(i + 1) + j * npatch_x)
                if j > 0:
                    p.set_neighbor_index(ymin=i + (j - 1) * npatch_x)
                if j < npatch_y - 1:
                    p.set_neighbor_index(ymax=i + (j + 1) * npatch_x)

                patches.append(p)

        def density(x, y):
            n0 = 2*nc

            return n0
        
        ele = Electron(density=density, ppc=8)
        ion = Proton(density=density, ppc=2)

        self.npart_ele = patches.add_species(ele)
        self.npart_ion = patches.add_species(ion)

        patches.fill_particles()
        
        for patch in patches:
            p = patch.particles[0]
            p.ux[:] = np.random.normal(0, 1, p.npart)
            p.uy[:] = np.random.normal(0, 1, p.npart)
            p.uz[:] = np.random.normal(0, 1, p.npart)
            p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2))**-0.5
            
        current_deposition_cpu(
            [patch.fields for patch in patches], 
            [patch.particles[0] for patch in patches],
            npatch_x*npatch_y, 1e-15, -e
        )