import unittest

import numpy as np
from scipy.constants import c

from libpic.fields import Fields2D
from libpic.patch import Patch2D, Patches
from libpic.qed.radiation import NonlinearComptonLCFA
from libpic.species import Electron, Photon


class TestNonlinearComptonLCFA(unittest.TestCase):
    def setUp(self) -> None:
        l0 = 0.8e-6
        nc = 1.74e27

        dx = 1e-6
        dy = 1e-6

        nx = 128
        ny = 128

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
        
        ele = Electron(density=density, ppc=1, radiation="photons")
        pho = Photon()

        ele.set_photon(pho)

        patches.add_species(ele)
        patches.add_species(pho)

        patches.fill_particles()
        
            
        for patch in patches:
            p = patch.particles[0]
            p.ux.fill(10)
            p.inv_gamma[:] = (1 + (p.ux**2 + p.uy**2 + p.uz**2))**-0.5
            p.chi.fill(0.1)
            p.is_dead[:] = np.random.uniform(size=p.is_dead.size) < 0.1
            
        self.patches = patches

        # 创建NonlinearComptonLCFA实例
        self.radiation = NonlinearComptonLCFA(self.patches, 0)

    def test_chi(self):
        self.radiation.generate_particle_lists()
        for patch in self.patches:
            p = patch.particles[0]
            p.ey_part.fill(1e12)
            p.chi.fill(0)

        chi = self.radiation.chi_list[0][0]
        self.assertEqual(chi, 0)

        self.radiation.update_chi()

        chi = self.radiation.chi_list[0][0]
        self.assertGreater(chi, 0)

    def test_event(self):
        self.radiation.generate_particle_lists()
        self.radiation.event(dt=0.1)
        print(self.radiation.event_list[0])

    def test_create_particles(self):
        self.radiation.generate_particle_lists()
        self.assertEqual(self.radiation.x_pho_list[0].size, 0)
        self.radiation.event(dt=0.1)
        self.radiation.create_particles()
        self.assertGreater(self.radiation.x_pho_list[0].size, 0)

    def test_reaction(self):
        self.radiation.generate_particle_lists()
        self.radiation.event(dt=0.1)
        ux = self.radiation.ux_list[0][0]
        self.assertEqual(ux, 10)
        self.radiation.reaction()
        ux = self.radiation.ux_list[0][0]
        delta = self.radiation.delta_list[0][0]
        self.assertEqual(ux, (1-delta)*10)

if __name__ == '__main__':
    unittest.main()
