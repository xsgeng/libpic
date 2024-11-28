import unittest

import numpy as np
from scipy.constants import c

from libpic.fields import Fields2D
from libpic.patch import Patch2D, Patches
from libpic.qed.pair_production import NonlinearPairProductionLCFA
from libpic.species import Electron, Photon, Positron


class TestNonlinearPairProductionLCFA(unittest.TestCase):
    def setUp(self) -> None:
        l0 = 0.8e-6
        nc = 1.74e27

        dx = 1e-6
        dy = 1e-6

        nx = 32
        ny = 32

        npatch_x = 2
        npatch_y = 2

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
            n0 = 0.01*nc
            return n0
        
        pho = Photon(density=density, ppc=1)
        ele = Electron()
        pos = Positron()

        pho.set_bw_pair(electron=ele, positron=pos)

        patches.add_species(pho)
        patches.add_species(ele)
        patches.add_species(pos)

        patches.fill_particles()

        patches.update_lists()
        
            
        for patch in patches:
            p = patch.particles[0]
            p.ux.fill(10)
            p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2))**-0.5
            p.chi.fill(0.1)
            p.is_dead[:] = np.random.uniform(size=p.is_dead.size) < 0.1
            
        self.patches = patches

        # 创建NonlinearPairProductionLCFA实例
        self.pair_production = NonlinearPairProductionLCFA(self.patches, 0)

    def test_chi(self):
        self.pair_production.generate_particle_lists()
        for patch in self.patches:
            p = patch.particles[0]
            p.ey_part.fill(1e12)
            p.chi.fill(0)

        chi = self.pair_production.chi_list[0][0]
        self.assertEqual(chi, 0)

        self.pair_production.update_chi()

        chi = self.pair_production.chi_list[0][0]
        self.assertGreater(chi, 0)

    def test_event(self):
        self.pair_production.generate_particle_lists()
        self.pair_production.event(dt=100)
        self.assertEqual(self.pair_production.event_list[0].sum(), np.logical_not(self.pair_production.is_dead_list[0]).sum())

    def test_create_particles(self):
        self.pair_production.generate_particle_lists()
        self.assertEqual(self.pair_production.x_ele_list[0].size, 0)
        self.assertEqual(self.pair_production.x_pos_list[0].size, 0)
        self.pair_production.event(dt=0.1)
        self.pair_production.create_particles()
        self.assertGreater(self.pair_production.x_ele_list[0].size, 0)
        self.assertGreater(self.pair_production.x_pos_list[0].size, 0)

    def test_reaction(self):
        self.pair_production.generate_particle_lists()
        self.pair_production.event(dt=0.1)
        is_alive = self.patches[0].particles[self.pair_production.ispec].is_alive
        self.assertGreater(is_alive.sum(), 0)

        self.pair_production.reaction()
        is_dead = self.pair_production.is_dead_list[0]
        self.assertTrue(is_dead.all())

if __name__ == '__main__':
    unittest.main()
