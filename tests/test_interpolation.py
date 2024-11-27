import unittest
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from libpic.interpolation.cpu import _interpolation_2d


class TestCurrentDeposition(unittest.TestCase):

    def test_interp2d(self):
        """test if interpolation_2d works"""
        npart = 3
        x = np.array([-2.0, np.nan, 1.0])
        y = np.full_like(x, 1.0)
        ex_part = np.full_like(x, 0)
        ey_part = np.full_like(x, 0)
        ez_part = np.full_like(x, 0)
        bx_part = np.full_like(x, 0)
        by_part = np.full_like(x, 0)
        bz_part = np.full_like(x, 0)
        is_dead = np.full(npart, False)
        is_dead[1] = True
        x[1] = np.nan

        nx = 3
        ny = 3
        x0 = 0.0
        y0 = 0.0
        dx = 1.0
        dy = 1.0
        
        ex = np.arange(nx*ny, dtype='f8').reshape((nx, ny))
        ey = ex
        ez = np.zeros((nx, ny))
        bx = np.zeros((nx, ny))
        by = np.zeros((nx, ny))
        bz = np.zeros((nx, ny))

        _interpolation_2d(
            x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, 
            is_dead,
            npart,
            ex, ey, ez, bx, by, bz,
            dx, dy, x0, y0, nx, ny,
        )
        self.assertSequenceEqual(ex_part.tolist(), [2.5, 0, 2.5])
        self.assertSequenceEqual(ey_part.tolist(), [3.5, 0, 3.5])

        