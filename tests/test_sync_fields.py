
import unittest
from time import perf_counter_ns

import numpy as np
from numba import typed
from scipy.constants import c, e


class TestPatches(unittest.TestCase):
    def test_sync_guard_2d(self):
        from libpic.patch import Patches, Patch2D
        from libpic.fields import Fields2D

        dx = 1.0
        dy = 1.0

        nx = 9
        ny = 9

        npatch_x = 3
        npatch_y = 3
        npatches = npatch_x * npatch_y

        nx_per_patch = nx//npatch_x
        ny_per_patch = ny//npatch_y

        Lx = nx*dx
        Ly = ny*dy

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
                f = Fields2D(nx=nx_per_patch, ny=ny_per_patch, dx=dx,dy=dy, x0=i*Lx/npatch_x, y0=j*Ly/npatch_y, n_guard=3)
                f.ex.fill(index)
                p.set_fields(f)

                # Edge neighbors
                if i > 0:
                    p.set_neighbor_index(xmin=(i - 1) + j * npatch_x)
                if i < npatch_x - 1:
                    p.set_neighbor_index(xmax=(i + 1) + j * npatch_x)
                if j > 0:
                    p.set_neighbor_index(ymin=i + (j - 1) * npatch_x)
                if j < npatch_y - 1:
                    p.set_neighbor_index(ymax=i + (j + 1) * npatch_x)
                # Corner neighbors
                if i > 0 and j > 0:
                    p.set_neighbor_index(xminymin=(i - 1) + (j - 1) * npatch_x)
                if i < npatch_x - 1 and j > 0:
                    p.set_neighbor_index(xmaxymin=(i + 1) + (j - 1) * npatch_x)
                if i > 0 and j < npatch_y - 1:
                    p.set_neighbor_index(xminymax=(i - 1) + (j + 1) * npatch_x)
                if i < npatch_x - 1 and j < npatch_y - 1:
                    p.set_neighbor_index(xmaxymax=(i + 1) + (j + 1) * npatch_x)

                patches.append(p)
        patches.update_lists()

        patches.sync_guard_fields()
        ex = patches[4].fields.ex

        n_guard = 3
        # center
        self.assertTrue((ex[:nx_per_patch, :ny_per_patch] == 4).all())
        # xmin = 3
        self.assertTrue((ex[-n_guard:, :ny_per_patch] == 3).all())
        # xmax = 5
        self.assertTrue((ex[nx_per_patch:nx_per_patch+n_guard, :ny_per_patch] == 5).all())
        # ymin = 1
        self.assertTrue((ex[:nx_per_patch, -n_guard:] == 1).all())
        # ymax = 7
        self.assertTrue((ex[:nx_per_patch, ny_per_patch:ny_per_patch+n_guard:] == 7).all())
        # xminymin = 0
        self.assertTrue((ex[-n_guard:, -n_guard:] == 0).all())
        # xmaxymin = 2
        self.assertTrue((ex[nx_per_patch:nx_per_patch+n_guard:, -n_guard:] == 2).all())
        # xminymax = 6
        self.assertTrue((ex[-n_guard:, ny_per_patch:ny_per_patch+n_guard:] == 6).all())
        # xmaxymax = 8
        self.assertTrue((ex[nx_per_patch:nx_per_patch+n_guard:, ny_per_patch:ny_per_patch+n_guard:] == 8).all())

    def test_sync_currents_2d(self):
        from libpic.patch import Patches, Patch2D
        from libpic.fields import Fields2D

        dx = 1.0
        dy = 1.0

        nx = 12
        ny = 12

        npatch_x = 3
        npatch_y = 3
        npatches = npatch_x * npatch_y

        nx_per_patch = nx//npatch_x
        ny_per_patch = ny//npatch_y

        Lx = nx*dx
        Ly = ny*dy

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
                f = Fields2D(nx=nx_per_patch, ny=ny_per_patch, dx=dx,dy=dy, x0=i*Lx/npatch_x, y0=j*Ly/npatch_y, n_guard=3)
                f.jx.fill(index)
                p.set_fields(f)

                # Edge neighbors
                if i > 0:
                    p.set_neighbor_index(xmin=(i - 1) + j * npatch_x)
                if i < npatch_x - 1:
                    p.set_neighbor_index(xmax=(i + 1) + j * npatch_x)
                if j > 0:
                    p.set_neighbor_index(ymin=i + (j - 1) * npatch_x)
                if j < npatch_y - 1:
                    p.set_neighbor_index(ymax=i + (j + 1) * npatch_x)
                # Corner neighbors
                if i > 0 and j > 0:
                    p.set_neighbor_index(xminymin=(i - 1) + (j - 1) * npatch_x)
                if i < npatch_x - 1 and j > 0:
                    p.set_neighbor_index(xmaxymin=(i + 1) + (j - 1) * npatch_x)
                if i > 0 and j < npatch_y - 1:
                    p.set_neighbor_index(xminymax=(i - 1) + (j + 1) * npatch_x)
                if i < npatch_x - 1 and j < npatch_y - 1:
                    p.set_neighbor_index(xmaxymax=(i + 1) + (j + 1) * npatch_x)

                patches.append(p)
        patches.update_lists()

        patches.sync_currents()
        jx = patches[4].fields.jx
        print('\n', jx)
        
        self.assertEqual((jx > 0).sum(), 16)

        #                          4+3+5+1+7+0+2+6+8
        self.assertEqual(jx[0, 0], 4+3+0+1+0+0+0+0+0)
        self.assertEqual(jx[0, 1], 4+3+0+1+7+0+0+6+0)
        self.assertEqual(jx[0, 2], 4+3+0+1+7+0+0+6+0)
        self.assertEqual(jx[0, 3], 4+3+0+0+7+0+0+6+0)

        self.assertEqual(jx[1, 0], 4+3+5+1+0+0+2+0+0)
        self.assertEqual(jx[1, 1], 4+3+5+1+7+0+2+6+8)
        self.assertEqual(jx[1, 2], 4+3+5+1+7+0+2+6+8)
        self.assertEqual(jx[1, 3], 4+3+5+0+7+0+0+6+8)

        self.assertEqual(jx[2, 0], 4+3+5+1+0+0+2+0+0)
        self.assertEqual(jx[2, 1], 4+3+5+1+7+0+2+6+8)
        self.assertEqual(jx[2, 2], 4+3+5+1+7+0+2+6+8)
        self.assertEqual(jx[2, 3], 4+3+5+0+7+0+0+6+8)

        self.assertEqual(jx[3, 0], 4+0+5+1+0+0+2+0+0)
        self.assertEqual(jx[3, 1], 4+0+5+1+7+0+2+0+8)
        self.assertEqual(jx[3, 2], 4+0+5+1+7+0+2+0+8)
        self.assertEqual(jx[3, 3], 4+0+5+0+7+0+0+0+8)