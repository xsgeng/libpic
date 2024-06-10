
import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.maxwell.solver import MaxwellSolver2d


class TestCurrentDeposition(unittest.TestCase):
    def test_solver_class(self):

        from libpic.fields import Fields2D
        from libpic.patch import Patch2D, Patches

        nx = 64
        ny = 64
        x0 = 0.0
        y0 = 0.0
        dx = 1.0e-6
        dy = 1.0e-6
        dt = dx / c / 2

        
        field = Fields2D(nx, ny, dx, dy, x0, y0, n_guard=3)
        patch = Patch2D(0, 0, 0, 0, x0, y0, field)
        patches = Patches()
        patches.append(patch)

        solver = MaxwellSolver2d(patches)

        solver.update_efield(dt)
        solver.update_bfield(dt)