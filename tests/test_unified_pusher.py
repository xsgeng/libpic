import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.pusher.unified.cpu import unified_boris_pusher_cpu


class TestCurrentDeposition(unittest.TestCase):
    def test_unified_pusher_class(self):
        from scipy.constants import c, pi
        from libpic.fields import Fields2D
        from libpic.patch.patch import Patch2D, Patches
        from libpic.species import Electron, Proton
        
        l0 = 0.8e-6
        nc = 1.74e27

        dx = 1e-6
        dy = 1e-6

        nx = 1280
        ny = 1280

        npatch_x = 16
        npatch_y = 16

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
        
        ele = Electron(density=density, ppc=100)

        self.npart_ele = patches.add_species(ele)

        patches.fill_particles()
        
        for patch in patches:
            p = patch.particles[0]
            p.ux[:] = np.random.normal(0, 1, p.npart)
            p.uy[:] = np.random.normal(0, 1, p.npart)
            p.uz[:] = np.random.normal(0, 1, p.npart)
            p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2))**-0.5
            
        tic = perf_counter_ns()
        unified_boris_pusher_cpu(
            [patch.particles[0] for patch in patches],
            [patch.fields for patch in patches], 
            npatch_x*npatch_y, 1e-15, ele.q, ele.m
        )
        toc = perf_counter_ns()

        npart = sum([patch.particles[0].npart for patch in patches])
        npatch = npatch_x*npatch_y
        import os
        nthreads = int(os.getenv('OMP_NUM_THREADS', os.cpu_count()))
        print(nthreads)
        nthreads = min(nthreads, npatch)
        print(f"unified_boris_pusher_cpu: {(toc - tic)/1e6} ms, {(toc - tic)/npart*nthreads:.0f} ns per particle")

if __name__ == "__main__":
    unittest.main()