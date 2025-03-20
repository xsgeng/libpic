import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.pusher.unified.unified_pusher_2d import unified_boris_pusher_cpu_2d
from libpic.pusher.unified.unified_pusher_3d import unified_boris_pusher_cpu_3d



class TestUnifiedPusher(unittest.TestCase):
    def test_2d_speed(self):
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
        unified_boris_pusher_cpu_2d(
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

    def test_3d_speed(self):
        from libpic.fields import Fields3D
        from libpic.patch.patch import Patch3D, Patches
        from libpic.species import Electron
        import os

        # Smaller 3D grid parameters
        dx = dy = dz = 1e-6
        nx = ny = nz = 128
        npatch_x = npatch_y = npatch_z = 16
        nx_per_patch = nx//npatch_x
        ny_per_patch = ny//npatch_y
        nz_per_patch = nz//npatch_z

        patches = Patches(dimension=3)
        for k in range(npatch_z):
            for j in range(npatch_y):
                for i in range(npatch_x):
                    index = i + j*npatch_x + k*npatch_x*npatch_y
                    p = Patch3D(
                        rank=0,
                        index=index,
                        ipatch_x=i,
                        ipatch_y=j,
                        ipatch_z=k,
                        x0=i*dx*nx_per_patch,
                        y0=j*dy*ny_per_patch,
                        z0=k*dz*nz_per_patch,
                        nx=nx_per_patch,
                        ny=ny_per_patch,
                        nz=nz_per_patch,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    )
                    f = Fields3D(
                        nx=nx_per_patch, ny=ny_per_patch, nz=nz_per_patch,
                        dx=dx, dy=dy, dz=dz,
                        x0=i*dx*nx_per_patch,
                        y0=j*dy*ny_per_patch,
                        z0=k*dz*nz_per_patch,
                        n_guard=3
                    )
                    p.set_fields(f)
                    
                    # Set neighbors in 3D
                    if i > 0: 
                        p.set_neighbor_index(xmin=(i-1) + j*npatch_x + k*npatch_x*npatch_y)
                    if i < npatch_x-1: 
                        p.set_neighbor_index(xmax=(i+1) + j*npatch_x + k*npatch_x*npatch_y)
                    if j > 0: 
                        p.set_neighbor_index(ymin=i + (j-1)*npatch_x + k*npatch_x*npatch_y)
                    if j < npatch_y-1: 
                        p.set_neighbor_index(ymax=i + (j+1)*npatch_x + k*npatch_x*npatch_y)
                    if k > 0: 
                        p.set_neighbor_index(zmin=i + j*npatch_x + (k-1)*npatch_x*npatch_y)
                    if k < npatch_z-1: 
                        p.set_neighbor_index(zmax=i + j*npatch_x + (k+1)*npatch_x*npatch_y)

                    patches.append(p)

        def density(x, y, z):  # 3D density function
            return 2*1.74e27

        ele = Electron(density=density, ppc=10)  # Fewer particles per cell
        patches.add_species(ele)
        patches.fill_particles()

        for patch in patches:
            p = patch.particles[0]
            # Initialize 3D velocities
            p.ux[:] = np.random.normal(0, 1, p.npart)
            p.uy[:] = np.random.normal(0, 1, p.npart)
            p.uz[:] = np.random.normal(0, 1, p.npart)
            p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2))**-0.5

        tic = perf_counter_ns()
        unified_boris_pusher_cpu_3d(  # 3D version of the pusher
            [patch.particles[0] for patch in patches],
            [patch.fields for patch in patches],
            npatch_x*npatch_y*npatch_z, 1e-15, ele.q, ele.m
        )
        toc = perf_counter_ns()

        npart = sum(patch.particles[0].npart for patch in patches)
        npatch = npatch_x*npatch_y*npatch_z
        nthreads = min(int(os.getenv('OMP_NUM_THREADS', os.cpu_count())), npatch)
        print(f"3D unified_boris_pusher_cpu: {(toc - tic)/1e6} ms, {(toc - tic)/npart*nthreads:.0f} ns per particle")

if __name__ == "__main__":
    unittest.main()
