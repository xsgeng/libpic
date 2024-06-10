
import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.boundary.cpml import PMLXmin, PMLXmax, PMLX, PMLY
from libpic.maxwell.solver import MaxwellSolver2d


class TestPML(unittest.TestCase):
    def setUp(self) -> None:
        from libpic.fields import Fields2D
        from libpic.patch import Patch2D, Patches

        nx = 100
        ny = 100
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
        patches = Patches()
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
                    fields=f,
                )

                if i > 0:
                    p.set_neighbor_index(xmin=(i - 1) + j * npatch_x)
                if i < npatch_x - 1:
                    p.set_neighbor_index(xmax=(i + 1) + j * npatch_x)
                if j > 0:
                    p.set_neighbor_index(ymin=i + (j - 1) * npatch_x)
                if j < npatch_y - 1:
                    p.set_neighbor_index(ymax=i + (j + 1) * npatch_x)

                if i == 0:
                    p.add_pml_boundary(PMLXmin(f))
                if i == npatch_x - 1:
                    p.add_pml_boundary(PMLXmax(f))

                patches.append(p)
        self.patches = patches
    
    def test_patches_pml(self):
        patches = self.patches

        self.assertEqual(len(patches[0].pml_boundary), 1)

        solver = MaxwellSolver2d(patches)
        solver.generate_field_lists()
        self.assertEqual(len(solver.ex_list), 24)

        solver.generate_kappa_lists()
        self.assertEqual(len(solver.kappa_ex_list), 8)

        solver.update_efield(self.dt)
        for p in patches:
            for pml in p.pml_boundary:
                pml.advance_e_currents(self.dt)
        solver.update_bfield(self.dt)
        for p in patches:
            for pml in p.pml_boundary:
                pml.advance_b_currents(self.dt)
