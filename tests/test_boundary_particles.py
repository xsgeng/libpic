
import unittest
from time import perf_counter_ns

import numpy as np
from numba import typed
from scipy.constants import c, e


# class TestBoundaryExchange(unittest.TestCase):
#     def test_count(self):
#         pos = np.zeros((2, 8))

#         pos[0, 0] -= 2
#         pos[0, 1] += 2

#         pos[1, 2] -= 2
#         pos[1, 3] += 2

#         pos[0, 4] -= 2
#         pos[1, 4] -= 2

#         pos[0, 5] += 2
#         pos[1, 5] -= 2

#         pos[0, 6] -= 2
#         pos[1, 6] += 2

#         pos[0, 7] += 2
#         pos[1, 7] += 2
#         outgoing = count_outgoing_particles(pos=pos, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)

#         self.assertTrue((outgoing == np.ones(8)).all())

#     def test_incoming_index(self):
#         pos_list = typed.List([np.zeros((2, 8)) for _ in range(9)])
#         xmin_index = 3
#         xmax_index = 5
#         ymin_index = 1
#         ymax_index = 7

#         xminymin_index = 0
#         xmaxymin_index = 2
#         xminymax_index = 6
#         xmaxymax_index = 8

#         dx = 1.0
#         dy = 1.0
#         xaxis_list = typed.List(
#             [
#                 np.arange(10, dtype='f8') + 0,
#                 np.arange(10, dtype='f8') + 10,
#                 np.arange(10, dtype='f8') + 20,
#                 np.arange(10, dtype='f8') + 0,
#                 np.arange(10, dtype='f8') + 10,
#                 np.arange(10, dtype='f8') + 20,
#                 np.arange(10, dtype='f8') + 0,
#                 np.arange(10, dtype='f8') + 10,
#                 np.arange(10, dtype='f8') + 20,
#             ]
#         )
#         yaxis_list = typed.List(
#             [
#                 np.arange(10, dtype='f8') + 0,
#                 np.arange(10, dtype='f8') + 0,
#                 np.arange(10, dtype='f8') + 0,
#                 np.arange(10, dtype='f8') + 10,
#                 np.arange(10, dtype='f8') + 10,
#                 np.arange(10, dtype='f8') + 10,
#                 np.arange(10, dtype=np.float64) + 20,
#                 np.arange(10, dtype='f8') + 20,
#                 np.arange(10, dtype='f8') + 20,
#             ]
#         )

#         xmin_indices = np.zeros(1, dtype='i8')
#         xmax_indices = np.zeros(1, dtype='i8')
#         ymin_indices = np.zeros(1, dtype='i8')
#         ymax_indices = np.zeros(1, dtype='i8')
#         xminymin_indices = np.zeros(1, dtype='i8')
#         xmaxymin_indices = np.zeros(1, dtype='i8')
#         xminymax_indices = np.zeros(1, dtype='i8')
#         xmaxymax_indices = np.zeros(1, dtype='i8')

#         x_base = np.array([-dx, 9+dx, 0, 0, -dx, 9+dx, -dx, 9+dx])
#         y_base = np.array([0, 0, -dy, 9+dy, -dy, -dy, 9+dy, 9+dy])
#         pos_list = typed.List(
#             [
#                 np.vstack([x_base + 0,  y_base + 0]),
#                 np.vstack([x_base + 10, y_base + 0]),
#                 np.vstack([x_base + 20, y_base + 0]),
               
#                 np.vstack([x_base + 0,  y_base + 10]),
#                 np.vstack([x_base + 10, y_base + 10]),
#                 np.vstack([x_base + 20, y_base + 10]),
               
#                 np.vstack([x_base + 0,  y_base + 20]),
#                 np.vstack([x_base + 10, y_base + 20]),
#                 np.vstack([x_base + 20, y_base + 20]),
#             ]
#         )

#         get_incoming_index(
#             pos_list=pos_list,
#             xmin_index=xmin_index,
#             xmax_index=xmax_index,
#             ymin_index=ymin_index,
#             ymax_index=ymax_index,
#             xminymin_index=xminymin_index,
#             xmaxymin_index=xmaxymin_index,
#             xminymax_index=xminymax_index,
#             xmaxymax_index=xmaxymax_index,
#             dx=dx,
#             dy=dy,
#             xaxis_list=xaxis_list,
#             yaxis_list=yaxis_list,
#             xmin_indices=xmin_indices,
#             xmax_indices=xmax_indices,
#             ymin_indices=ymin_indices,
#             ymax_indices=ymax_indices,
#             xminymin_indices=xminymin_indices,
#             xmaxymin_indices=xmaxymin_indices,
#             xminymax_indices=xminymax_indices,
#             xmaxymax_indices=xmaxymax_indices,
#         )
#         self.assertTrue(xmax_indices == 0)
#         self.assertTrue(xmin_indices == 1)
#         self.assertTrue(ymax_indices == 2)
#         self.assertTrue(ymin_indices == 3)
#         self.assertTrue(xmaxymax_indices == 4)
#         self.assertTrue(xminymax_indices == 5)
#         self.assertTrue(xmaxymin_indices == 6)
#         self.assertTrue(xminymin_indices == 7)

#         buffer = np.zeros((8, 2)) 
#         fill_boundary_particles_to_buffer(
#             attrs_list=(pos_list, ),
#             xmin_indices=xmin_indices,
#             xmax_indices=xmax_indices,
#             ymin_indices=ymin_indices,
#             ymax_indices=ymax_indices,
#             xminymin_indices=xminymin_indices,
#             xmaxymin_indices=xmaxymin_indices,
#             xminymax_indices=xminymax_indices,
#             xmaxymax_indices=xmaxymax_indices,
#             xmin_index=xmin_index,
#             xmax_index=xmax_index,
#             ymin_index=ymin_index,
#             ymax_index=ymax_index,
#             xminymin_index=xminymin_index,
#             xmaxymin_index=xmaxymin_index,
#             xminymax_index=xminymax_index,
#             xmaxymax_index=xmaxymax_index,
#             buffer=buffer,
#         )
#         self.assertTrue(all(buffer[0] == [10, 10]), f"xmin not correct")
#         self.assertTrue(all(buffer[1] == [19, 10]), f"xmax not correct")
#         self.assertTrue(all(buffer[2] == [10, 10]), f"ymin not correct")
#         self.assertTrue(all(buffer[3] == [10, 19]), f"ymax not correct")
#         self.assertTrue(all(buffer[4] == [10, 10]), f"xminymin not correct")
#         self.assertTrue(all(buffer[5] == [19, 10]), f"xmaxymin not correct")
#         self.assertTrue(all(buffer[6] == [10, 19]), f"xminymax not correct")
#         self.assertTrue(all(buffer[7] == [19, 19]), f"xmaxymax not correct")

class TestPatches(unittest.TestCase):
    def test_2d(self):
        from libpic.patch import Patches, Patch2D
        from libpic.species import Electron
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
                p.set_fields(f)

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

        
        ele = Electron(density=lambda x, y : 1.0, ppc=1)

        patches.add_species(ele)
        patches.update_lists()

        patches.fill_particles()
        patches[0].particles[0].x[:] += dx
        patches[0].particles[0].y[:] += dy

        x_out = patches[0].particles[0].x[6:8].tolist()
        y_out = [patches[0].particles[0].y[2], patches[0].particles[0].y[5]]
        corner_out = patches[0].particles[0].x[8]

        patches.sync_particles()

        self.assertListEqual(x_out, patches[1].particles[0].x[9:11].tolist())
        self.assertListEqual(y_out, patches[3].particles[0].y[9:11].tolist())
        self.assertEqual(corner_out, patches[4].particles[0].x[9])

        for patch in patches:
            p = patch.particles[0]
            alive = np.logical_not(p.is_dead)
            with self.subTest(f"Patch {patch.index}:"):
                self.assertTrue(all(~np.isnan(p.x[alive])), f"pos={p.x} is_dead={p.is_dead}")

    def test_3d(self):
        from libpic.patch import Patches, Patch3D
        from libpic.species import Electron
        from libpic.fields import Fields3D


        dx = 1.0
        dy = 1.0
        dz = 1.0

        nx = 9
        ny = 9
        nz = 9

        npatch_x = 3
        npatch_y = 3
        npatch_z = 3
        npatches = npatch_x * npatch_y * npatch_z

        nx_per_patch = nx//npatch_x
        ny_per_patch = ny//npatch_y
        nz_per_patch = nz//npatch_z

        Lx = nx*dx
        Ly = ny*dy

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
                    patches.append(p)
        
        patches.init_rect_neighbor_index_3d(npatch_x, npatch_y, npatch_z)

        
        ele = Electron(density=lambda x, y, z : 1.0, ppc=1)

        patches.add_species(ele)
        patches.update_lists()

        patches.fill_particles()

        particles = patches[0].particles[0]
        particles.x[:] += dx
        particles.y[:] += dy
        particles.z[:] += dz

        particles = patches[13].particles[0]
        particles.x[:] -= dx
        particles.y[:] -= dy
        particles.z[:] -= dz

        patches.sync_particles()

        npart_total = sum([patch.particles[0].is_alive.sum() for patch in patches])
        self.assertEqual(npart_total, nx*ny*nz)