from numba import float64, typed, types, njit, prange, set_num_threads
from numba.extending import as_numba_type
import numpy as np

from time import perf_counter_ns

from maxwell_2d import update_bfield_2d, update_efield_2d
from fields import Fields2D
from particles import Particles

class Patch2D:
    def __init__(
            self,
            rank: int,
            index: int,
            ipatch_x: int,
            ipatch_y: int,
            x0 : float, 
            y0 : float,
        ) -> None:
        self.rank = rank
        self.index = index
        self.ipatch_x = ipatch_x
        self.ipatch_y = ipatch_y
        self.x0 = x0
        self.y0 = y0

        # neighbors
        self.xmin_neighbor_index : int = None
        self.xmax_neighbor_index : int = None
        self.ymin_neighbor_index : int = None
        self.ymax_neighbor_index : int = None
        # MPI neighbors
        self.xmin_neighbor_rank : int = None
        self.xmax_neighbor_rank : int = None
        self.ymin_neighbor_rank : int = None
        self.ymax_neighbor_rank : int = None
    
    def set_neighbor_index(self, *, xmin : int=None, xmax : int=None, ymin : int=None, ymax : int=None):
        if xmin is not None:
            self.xmin_neighbor_index = xmin
        if xmax is not None:
            self.xmax_neighbor_index = xmax
        if ymin is not None:
            self.ymin_neighbor_index = ymin
        if ymax is not None:
            self.ymax_neighbor_index = ymax

    def set_neighbor_rank(self, *, xmin : int=None, xmax : int=None, ymin : int=None, ymax : int=None):
        if xmin is not None:
            self.xmin_neighbor_rank = xmin
        if xmax is not None:
            self.xmax_neighbor_rank = xmax
        if ymin is not None:
            self.ymin_neighbor_rank = ymin
        if ymax is not None:
            self.ymax_neighbor_rank = ymax


    def set_fields(self, fields: Fields2D):
        self.fields = fields

    def set_particles(self, particles: Particles):
        self.particles = particles

class Patches2D:
    """ 
    A container for patches of the fields and particles. 
    The patches will be created by the main class.
    """
    def __init__(self) -> None:
        self.npatches = 0
        self.indexs : list[int] = []
        self.patches : list[Patch2D] = []
    
    def __getitem__(self, i: int) -> Patch2D:
        return self.patches[i]

    def __len__(self) -> int:
        return self.npatches

    def __iter__(self):
        return iter(self.patches)

    def append(self, patch: Patch2D):
        if self.patches:
            assert self.patches[-1].index == patch.index - 1
        self.patches.append(patch)
        self.indexs.append(patch.index)
        self.npatches += 1

    def prepend(self, patch: Patch2D):
        if self.patches:
            assert self.patches[0].index == patch.index + 1
        self.patches.insert(0, patch)
        self.indexs.insert(0, patch.index)
        self.npatches += 1
    
    def pop(self, i):
        self.indexs.pop(i)
        self.npatches -= 1
        return self.patches.pop(i)


    def sync_guard_fields(self):
        ng = self[0].fields.n_guard
        for p in self.patches:
            if p.xmin_neighbor_index is not None:
                p.fields[-ng:, :] = self.patches[p.xmin_neighbor_index].fields[-3*ng:-2*ng, :]
            if p.xmax_neighbor_index is not None:
                p.fields[-2*ng:-ng, :] = self.patches[p.xmax_neighbor_index].fields[:ng, :]
            if p.ymin_neighbor_index is not None:
                p.fields[:, -ng:] = self.patches[p.ymin_neighbor_index].fields[:, -3*ng:-2*ng]
            if p.ymax_neighbor_index is not None:
                p.fields[:, -2*ng:-ng] = self.patches[p.ymax_neighbor_index].fields[:, :ng]

    def init_fields(self, nx, ny):
        self.fields = [Fields2D(nx, ny, 6) for _ in range(self.npatches)]

    def init_particles(self, npart):
        self.particles = [Particles(npart) for _ in range(self.npatches)]

    @property
    def nx(self):
        return self[0].fields.nx
    @property
    def ny(self):
        return self[0].fields.ny
    @property
    def dx(self):
        return self[0].fields.dx
    @property
    def dy(self):
        return self[0].fields.dy

    @property
    def n_guard(self):
        return self[0].fields.n_guard

    def update_efield(self, dt):
        update_efield_patches(
            ex_list = typed.List([p.fields.ex for p in self.patches]),
            ey_list = typed.List([p.fields.ey for p in self.patches]),
            ez_list = typed.List([p.fields.ez for p in self.patches]),
            bx_list = typed.List([p.fields.bx for p in self.patches]),
            by_list = typed.List([p.fields.by for p in self.patches]),
            bz_list = typed.List([p.fields.bz for p in self.patches]),
            jx_list = typed.List([p.fields.jx for p in self.patches]),
            jy_list = typed.List([p.fields.jy for p in self.patches]),
            jz_list = typed.List([p.fields.jz for p in self.patches]),
            npatches = self.npatches, 
            dx = self.dx, 
            dy = self.dy, 
            dt = dt, 
            nx = self.nx,
            ny = self.ny, 
            n_guard = self.n_guard,
        )

    def update_bfield(self, dt):
        update_bfield_patches(
            ex_list = typed.List([p.fields.ex for p in self.patches]),
            ey_list = typed.List([p.fields.ey for p in self.patches]),
            ez_list = typed.List([p.fields.ez for p in self.patches]),
            bx_list = typed.List([p.fields.bx for p in self.patches]),
            by_list = typed.List([p.fields.by for p in self.patches]),
            bz_list = typed.List([p.fields.bz for p in self.patches]),
            npatches = self.npatches, 
            dx = self.dx, 
            dy = self.dy, 
            dt = dt, 
            nx = self.nx,
            ny = self.ny, 
            n_guard = self.n_guard,
        )

@njit(parallel=True)
def update_efield_patches(
    ex_list, ey_list, ez_list, 
    bx_list, by_list, bz_list, 
    jx_list, jy_list, jz_list, 
    npatches,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for i in prange(npatches):
        ex = ex_list[i]
        ey = ey_list[i]
        ez = ez_list[i]
        bx = bx_list[i]
        by = by_list[i]
        bz = bz_list[i]
        jx = jx_list[i]
        jy = jy_list[i]
        jz = jz_list[i]

        update_efield_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, dt, nx, ny, n_guard)

@njit(parallel=True)
def update_bfield_patches(
    ex_list, ey_list, ez_list, 
    bx_list, by_list, bz_list, 
    npatches,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for i in prange(npatches):
        ex = ex_list[i]
        ey = ey_list[i]
        ez = ez_list[i]
        bx = bx_list[i]
        by = by_list[i]
        bz = bz_list[i]

        update_bfield_2d(ex, ey, ez, bx, by, bz, dx, dy, dt, nx, ny, n_guard)
