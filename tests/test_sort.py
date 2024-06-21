import unittest
import numpy as np
from libpic.sort.cpu import sorted_cell_bound

import numpy as np
import unittest

from libpic.sort.cpu import calculate_cell_index

class TestSortParticles(unittest.TestCase): 
    def setUp(self):
        self.nx = 5
        self.ny = 7
        self.dx = 1.0
        self.dy = 1.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.num_particles = 1000
        self.x = np.random.uniform(self.x0, self.x0 + self.nx * self.dx, self.num_particles)
        self.y = np.random.uniform(self.y0, self.y0 + self.ny * self.dy, self.num_particles)
        self.ux = np.random.uniform(-1, 1, self.num_particles)
        self.uy = np.random.uniform(-1, 1, self.num_particles)
        self.uz = np.random.uniform(-1, 1, self.num_particles)
        self.particle_cell_indices = np.zeros(self.num_particles, dtype=int)
        self.grid_cell_count = np.zeros((self.nx, self.ny), dtype=int)
        self.cell_bound_min = np.zeros((self.nx, self.ny), dtype=int)
        self.cell_bound_max = np.zeros((self.nx, self.ny), dtype=int)

    def test_calculate_cell_index(self):
        calculate_cell_index(self.x, self.y, self.nx, self.ny, self.dx, self.dy, self.x0, self.y0, self.particle_cell_indices, self.grid_cell_count)
        
        particle_cell_indices_expected = np.floor((self.y - self.y0) / self.dy).astype(int) + np.floor((self.x - self.x0) / self.dx).astype(int) * self.ny
        grid_cell_count_expected = np.histogram2d(self.x, self.y, bins=(self.nx, self.ny), range=((self.x0, self.x0 + self.nx * self.dx), (self.y0, self.y0 + self.ny * self.dy)))[0]
        self.assertTrue(np.array_equal(particle_cell_indices_expected, self.particle_cell_indices))
        self.assertTrue((grid_cell_count_expected==self.grid_cell_count).all())


    def test_cell_sort(self):
        calculate_cell_index(self.x, self.y, self.nx, self.ny, self.dx, self.dy, self.x0, self.y0, self.particle_cell_indices, self.grid_cell_count)
        sorted_cell_bound(self.grid_cell_count, self.cell_bound_min, self.cell_bound_max, self.nx, self.ny)

        self.assertEqual(self.cell_bound_max[-1, -1], self.num_particles)
        for ix in range(self.nx):
            for iy in range(self.ny):
                cell_start = self.cell_bound_min[ix, iy]
                cell_end = self.cell_bound_max[ix, iy]
                self.assertLessEqual(cell_start, cell_end)
        #         for j in range(cell_start, cell_end):
        #             self.assertTrue((self.x[j] >= self.x0 + ix * self.dx) and (self.x[j] < self.x0 + (ix + 1) * self.dx))
        #             self.assertTrue((self.y[j] >= self.y0 + iy * self.dy) and (self.y[j] < self.y0 + (iy + 1) * self.dy))

class TestPatches(unittest.TestCase):
    def setUp(self) -> None:
        from libpic.patch.patch import Patches, Patch2D
        from libpic.fields import Fields2D
        from scipy.constants import pi, c
        from libpic.species import Electron, Proton
        
        l0 = 0.8e-6
        nc = 1.74e27

        dx = 1e-6
        dy = 1e-6

        nx = 100
        ny = 120

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
            r2 = (x-Lx/2)**2 + (y-Ly/2)**2
            r = r2**0.5
            ne = 0.0

            n0 = 2*nc

            return n0
        
        ele = Electron(density=density, ppc=2)
        ion = Proton(density=density, ppc=2)

        self.npart = 0
        self.npart += patches.add_species(ele)
        self.npart += patches.add_species(ion)

        patches.fill_particles()
        
            
        for patch in patches:
            p = patch.particles[0]
            p.ux[:] = np.random.normal(0, 1, p.npart)
            p.uy[:] = np.random.normal(0, 1, p.npart)
            p.uz[:] = np.random.normal(0, 1, p.npart)
            p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2))**-0.5
            idx = np.random.permutation(p.x.size)
            p.x = p.x[idx]
            p.y = p.y[idx]
            
        self.patches = patches
        
    def test_init_sort(self):
        from libpic.sort.particle_sort import ParticleSort2D
        from libpic.util import Timer
        sorter = ParticleSort2D(self.patches, 0)
        
        
        # sorter.calculate_cell_index()
        # sorter.cell_sort()
        # with Timer('sorter per particle', norm=self.npart, unit='ns'):
        #     with Timer('sorter', norm=1, unit='ms'):
        #         sorter()
        sorter()
        for ipatch, patch in enumerate(self.patches):
            p = patch.particles[0]
            for ix in range(patch.nx):
                for iy in range(patch.ny):
                    cell_start = sorter.cell_bound_min_list[ipatch][ix, iy]
                    cell_end = sorter.cell_bound_max_list[ipatch][ix, iy]
                    for ip in range(cell_start, cell_end):
                        self.assertGreaterEqual(p.x[ip], sorter.x0s[ipatch] + ix * sorter.dx)
                        self.assertLess(p.x[ip], sorter.x0s[ipatch] + ( ix + 1 ) * sorter.dx)
                        
                        self.assertGreaterEqual(p.y[ip], sorter.y0s[ipatch] + iy * sorter.dy)
                        self.assertLess(p.y[ip], sorter.y0s[ipatch] + ( iy + 1 ) * sorter.dy)

        
if __name__ == '__main__':
   unittest.main()
