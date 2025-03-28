
import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.boundary.cpml import PMLXmin, PMLXmax, PMLX, PMLY, PMLYmin, PMLYmax, PMLZ, PMLZmin, PMLZmax
from libpic.maxwell.solver import MaxwellSolver2D


class TestPML2D(unittest.TestCase):
    def setUp(self) -> None:
        from libpic.fields import Fields2D
        from libpic.patch import Patch2D, Patches

        nx = 100
        ny = 90
        x0 = 0.0
        y0 = 0.0
        dx = 1.0e-6
        dy = 1.0e-6
        self.dt = dx / c / 2

        self.field = Fields2D(nx, ny, dx, dy, x0, y0, n_guard=3)
        
    def test_xmin(self):
        pml = PMLXmin(self.field)
        self.assertEqual(pml.thickness, 6)
        # print(pml.sigma_ex, pml.sigma_ey)
        # print(pml.sigma_bx, pml.sigma_by)

        pml.advance_e_currents(self.dt)
        tic = perf_counter_ns()
        for _ in range(100): pml.advance_e_currents(self.dt)
        toc = perf_counter_ns()
        self.assertLess(toc - tic, 2e6)

    def test_xmax(self):
        pml = PMLXmax(self.field)
        self.assertIsInstance(pml, PMLX)
        self.assertEqual(pml.nx, pml.fields.ex.shape[0]-6)
        self.assertEqual(pml.ny, pml.fields.ex.shape[1]-6)
        self.assertEqual(pml.thickness, 6)
        # print(pml.sigma_ex, pml.sigma_ey)
        # print(pml.sigma_bx, pml.sigma_by)

        pml.advance_e_currents(self.dt)
        tic = perf_counter_ns()
        for _ in range(100): pml.advance_e_currents(self.dt)
        toc = perf_counter_ns()
        self.assertLess(toc - tic, 2e6)

class TestPML3D(unittest.TestCase):
    def setUp(self) -> None:
        from libpic.fields import Fields3D

        # Use different dimensions to catch slicing errors
        self.nx, self.ny, self.nz = 100, 90, 80  
        dx = dy = dz = 1e-6
        self.dt = dx / c / 2

        self.field = Fields3D(self.nx, self.ny, self.nz, dx, dy, dz, 
                             0, 0, 0, n_guard=3)
        
        # Initialize fields with non-zero values to test PML effects
        self.field.bx[:, :, :] = 1.0
        self.field.by[:, :, :] = 1.0
        self.field.bz[:, :, :] = 1.0

    def _common_pml_checks(self, pml):
        """Common assertions for all PML tests"""
        self.assertEqual(pml.thickness, 6)
        
    def test_xmin(self):
        pml = PMLXmin(self.field)
        self._common_pml_checks(pml)
        self.assertEqual(pml.psi_ey_x.shape, (self.nx, self.ny, self.nz))
        
        # Check PML region parameters
        self.assertTrue(np.all(pml.sigma_ex[:6] > 0))
        self.assertTrue(np.all(pml.kappa_ex[:6] > 1))
        
        pml.advance_e_currents(self.dt)

    def test_xmax(self):
        pml = PMLXmax(self.field)
        self._common_pml_checks(pml)
        self.assertEqual(pml.psi_ey_x.shape, (self.nx, self.ny, self.nz))
        self.assertTrue(np.all(pml.sigma_ex[-6:] > 0))

        # Test both current advances
        pml.advance_e_currents(self.dt)
        pml.advance_b_currents(self.dt)
        

    def test_ymin(self):
        pml = PMLYmin(self.field)
        self._common_pml_checks(pml)
        self.assertEqual(pml.psi_ex_y.shape, (self.nx, self.ny, self.nz))
        self.assertTrue(np.all(pml.sigma_ey[:6] > 0))
        
        # Test both current advances
        pml.advance_e_currents(self.dt)
        pml.advance_b_currents(self.dt)

    def test_ymax(self):
        pml = PMLYmax(self.field)
        self._common_pml_checks(pml)
        self.assertEqual(pml.psi_ex_y.shape, (self.nx, self.ny, self.nz))
        self.assertTrue(np.all(pml.sigma_ey[-6:] > 0))

        # Test both current advances
        pml.advance_e_currents(self.dt)
        pml.advance_b_currents(self.dt)

    def test_zmin(self):
        pml = PMLZmin(self.field)
        self._common_pml_checks(pml)
        self.assertEqual(pml.psi_ex_z.shape, (self.nx, self.ny, self.nz))
        self.assertTrue(np.all(pml.sigma_ez[:6] > 0))

        # Test both current advances
        pml.advance_e_currents(self.dt)
        pml.advance_b_currents(self.dt)
        

    def test_zmax(self):
        pml = PMLZmax(self.field)
        self._common_pml_checks(pml)
        self.assertEqual(pml.psi_ex_z.shape, (self.nx, self.ny, self.nz))
        self.assertTrue(np.all(pml.sigma_ez[-6:] > 0))
        
        # Test both current advances
        pml.advance_e_currents(self.dt)
        pml.advance_b_currents(self.dt)

class TestPatchesPML(unittest.TestCase):
    def setUp(self) -> None:
        from libpic.fields import Fields2D
        from libpic.patch import Patch2D, Patches

        dx = 1e-8
        dy = 1e-8

        self.dt = dx / c / 2

        nx = 128
        ny = 128

        npatch_x = 8
        npatch_y = 4
        npatches = npatch_x * npatch_y

        nx_per_patch = nx//npatch_x
        ny_per_patch = ny//npatch_y

        Lx = nx*dx
        Ly = ny*dy

        n_guard = 3
        patches = Patches(dimension=2)
        for j in range(npatch_y):
            for i in range(npatch_x):
                index = i + j * npatch_x
                
                f = Fields2D(nx=nx_per_patch, ny=ny_per_patch, dx=dx,dy=dy, x0=i*Lx/npatch_x, y0=j*Ly/npatch_y, n_guard=n_guard)

                p = Patch2D(
                    rank=0, 
                    index=index, 
                    ipatch_x=i, 
                    ipatch_y=j, 
                    x0=i*Lx/npatch_x, 
                    y0=j*Ly/npatch_y,
                    nx=nx_per_patch, 
                    ny=ny_per_patch,
                    dx=dx, dy=dy
                )
                p.set_fields(f)

                if i == 0:
                    p.add_pml_boundary(PMLXmin(f))
                if i == npatch_x - 1:
                    p.add_pml_boundary(PMLXmax(f))
                if j == 0:
                    p.add_pml_boundary(PMLYmin(f))
                if j == npatch_y - 1:
                    p.add_pml_boundary(PMLYmax(f))

                patches.append(p)
        patches.init_rect_neighbor_index_2d(npatch_x, npatch_y)
        self.patches = patches
        self.npatch_x = npatch_x
        self.npatch_y = npatch_y
    
    def test_patches_pml(self):
        patches = self.patches

        self.assertEqual(len(patches[0].pml_boundary), 2)

        solver = MaxwellSolver2D(patches)
        solver.generate_field_lists()
        self.assertEqual(len(solver.ex_list), (self.npatch_x-2) * (self.npatch_y-2))

        solver.generate_kappa_lists()
        self.assertEqual(len(solver.kappa_ex_list), 2*self.npatch_x + 2*(self.npatch_y - 2))

        solver.update_efield(self.dt)
        solver.update_bfield(self.dt)
