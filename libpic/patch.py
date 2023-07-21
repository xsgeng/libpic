from numba import float64, typed, types, njit, prange, set_num_threads
from numba.extending import as_numba_type
import numpy as np

from time import perf_counter_ns

from .maxwell_2d import update_bfield_2d, update_efield_2d
from .fields import Fields2D
from .particles import Particles
from .species import Species

class Patch2D:
    def __init__(
        self,
        rank: int,
        index: int,
        ipatch_x: int,
        ipatch_y: int,
        x0 : float, 
        y0 : float,
        nx : int,
        ny : int,
        dx : float, 
        dy : float,
    ) -> None:
        """ 
        Patch2D is a container for the fields and particles of a single patch.
        The patch is a rectangular region of the grid.

        Parameters
        ----------
        rank : int
            rank of the process
        index : int
            index of the patch
        ipatch_x : int
            index of the patch in x direction
        ipatch_y : int
            index of the patch in y direction
        x0 : float
            start x0 of the patch
        y0 : float
            start y0 of the patch
        nx : int
            number of grids in x direction of the patch, n_guard not included
        ny : int
            number of grids in y direction of the patch, n_guard not included
        dx : float
            grid spacing in x direction
        dy : float
            grid spacing in y direction
        """
        self.rank = rank
        self.index = index
        self.ipatch_x = ipatch_x
        self.ipatch_y = ipatch_y
        self.x0 = x0
        self.y0 = y0
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        self.xaxis = np.arange(nx) * dx + x0
        self.yaxis = np.arange(ny) * dy + y0

        # neighbors
        self.xmin_neighbor_index : int = -1
        self.xmax_neighbor_index : int = -1
        self.ymin_neighbor_index : int = -1
        self.ymax_neighbor_index : int = -1
        # MPI neighbors
        self.xmin_neighbor_rank : int = -1
        self.xmax_neighbor_rank : int = -1
        self.ymin_neighbor_rank : int = -1
        self.ymax_neighbor_rank : int = -1
    
    def set_neighbor_index(self, *, xmin : int=-1, xmax : int=-1, ymin : int=-1, ymax : int=-1):
        if xmin >= 0:
            self.xmin_neighbor_index = xmin
        if xmax >= 0:
            self.xmax_neighbor_index = xmax
        if ymin >= 0:
            self.ymin_neighbor_index = ymin
        if ymax >= 0:
            self.ymax_neighbor_index = ymax

    def set_neighbor_rank(self, *, xmin : int=-1, xmax : int=-1, ymin : int=-1, ymax : int=-1):
        if xmin >= 0:
            self.xmin_neighbor_rank = xmin
        if xmax >= 0:
            self.xmax_neighbor_rank = xmax
        if ymin >= 0:
            self.ymin_neighbor_rank = ymin
        if ymax >= 0:
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
        p = self.patches.pop(i)

        return p

    def update_lists(self):
        """ this update is expensive. """
        self.ex_list : typed.List = typed.List([p.fields.ex for p in self.patches])
        self.ey_list : typed.List = typed.List([p.fields.ey for p in self.patches])
        self.ez_list : typed.List = typed.List([p.fields.ez for p in self.patches])
        self.bx_list : typed.List = typed.List([p.fields.bx for p in self.patches])
        self.by_list : typed.List = typed.List([p.fields.by for p in self.patches])
        self.bz_list : typed.List = typed.List([p.fields.bz for p in self.patches])
        self.jx_list : typed.List = typed.List([p.fields.jx for p in self.patches])
        self.jy_list : typed.List = typed.List([p.fields.jy for p in self.patches])
        self.jz_list : typed.List = typed.List([p.fields.jz for p in self.patches])

        self.xaxis_list : typed.List = typed.List([p.xaxis for p in self.patches])
        self.yaxis_list : typed.List = typed.List([p.yaxis for p in self.patches])

        self.xmin_neighbor_index_list = typed.List([p.xmin_neighbor_index for p in self.patches])
        self.xmax_neighbor_index_list = typed.List([p.xmax_neighbor_index for p in self.patches])
        self.ymin_neighbor_index_list = typed.List([p.ymin_neighbor_index for p in self.patches])
        self.ymax_neighbor_index_list = typed.List([p.ymax_neighbor_index for p in self.patches])

        self.x_list = typed.List([p.particles.x for p in self.patches])
        self.y_list = typed.List([p.particles.y for p in self.patches])
        self.w_list = typed.List([p.particles.w for p in self.patches])
        self.ux_list = typed.List([p.particles.ux for p in self.patches])
        self.uy_list = typed.List([p.particles.uy for p in self.patches])
        self.uz_list = typed.List([p.particles.uz for p in self.patches])

    def sync_guard_fields(self):
        sync_guard_fields(
            self.ex_list,
            self.ey_list,
            self.ez_list,
            self.bx_list,
            self.by_list,
            self.bz_list,
            self.jx_list,
            self.jy_list,
            self.jz_list,
            self.xmin_neighbor_index_list, 
            self.xmax_neighbor_index_list, 
            self.ymin_neighbor_index_list, 
            self.ymax_neighbor_index_list, 
            self.npatches, 
            self.nx,
            self.ny,
            self.n_guard,
        )
        
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
            ex_list = self.ex_list,
            ey_list = self.ey_list,
            ez_list = self.ez_list,
            bx_list = self.bx_list,
            by_list = self.by_list,
            bz_list = self.bz_list,
            jx_list = self.jx_list,
            jy_list = self.jy_list,
            jz_list = self.jz_list,
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
            ex_list = self.ex_list, 
            ey_list = self.ey_list, 
            ez_list = self.ez_list, 
            bx_list = self.bx_list, 
            by_list = self.by_list, 
            bz_list = self.bz_list, 
            npatches = self.npatches, 
            dx = self.dx, 
            dy = self.dy, 
            dt = dt, 
            nx = self.nx,
            ny = self.ny, 
            n_guard = self.n_guard,
        )

    def init_particles(self, species : Species):

        self.xaxis_list : typed.List = typed.List([p.xaxis for p in self.patches])
        self.yaxis_list : typed.List = typed.List([p.yaxis for p in self.patches])
        self.density_func = njit(species.density)

        num_macro_particles = get_num_macro_particles(
            self.density_func,
            self.xaxis_list, 
            self.yaxis_list, 
            self.npatches, 
            species.density_min, 
            species.ppc,
        )

        for ipatch in range(self.npatches):
            particles : Particles = species.create_particles()
            particles.initialize(num_macro_particles[ipatch])
            self[ipatch].set_particles(particles)

    def fill_particles(self, species : Species):
        self.x_list = typed.List([p.particles.x for p in self.patches])
        self.y_list = typed.List([p.particles.y for p in self.patches])
        self.w_list = typed.List([p.particles.w for p in self.patches])
        fill_particles(
            self.density_func,
            self.xaxis_list, 
            self.yaxis_list, 
            self.npatches, 
            species.density_min, 
            species.ppc,
            self.x_list, self.y_list, self.w_list
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

@njit(parallel=True)
def sync_guard_fields(
    ex_list, ey_list, ez_list, 
    bx_list, by_list, bz_list, 
    jx_list, jy_list, jz_list, 
    xmin_index_list, 
    xmax_index_list, 
    ymin_index_list, 
    ymax_index_list, 
    npatches, nx, ny, ng
):
    all_fields = [ex_list, ey_list, ez_list, bx_list, by_list, bz_list, jx_list, jy_list, jz_list]
    for i in prange(npatches*9):
        field = all_fields[i%9]
        ipatch = i//9
        xmin_index = xmin_index_list[ipatch]
        xmax_index = xmax_index_list[ipatch]
        ymin_index = ymin_index_list[ipatch]
        ymax_index = ymax_index_list[ipatch]
        if xmin_index >= 0:
            field[ipatch][-ng:, :ny] = field[xmin_index][nx-ng:nx, :ny]
        if ymin_index >= 0:
            field[ipatch][:nx, -ng:] = field[ymin_index][:ny, ny-ng:ny]
        if xmax_index >= 0:
            field[ipatch][nx:nx+ng, :ny] = field[xmax_index][:ng, :ny]
        if ymax_index >= 0:
            field[ipatch][:nx, nx:nx+ng] = field[ymax_index][:nx, :ng]


@njit(parallel=True)
def get_num_macro_particles(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc) -> np.ndarray:
    num_particles = np.zeros(npatches, dtype=np.int64)
    for ipatch in prange(npatches):
        xaxis =  xaxis_list[ipatch]
        yaxis =  yaxis_list[ipatch]
        
        for x_grid in xaxis:
            for y_grid in yaxis:
                dens = density_func(x_grid, y_grid)
                if dens > dens_min:
                    num_particles[ipatch] += ppc
    return num_particles

@njit(parallel=True)
def fill_particles(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc, x_list, y_list, w_list):
    dx = xaxis_list[0][1] - xaxis_list[0][0]
    dy = yaxis_list[0][1] - yaxis_list[0][0]
    for ipatch in prange(npatches):
        xaxis =  xaxis_list[ipatch]
        yaxis =  yaxis_list[ipatch]
        x = x_list[ipatch]
        y = y_list[ipatch]
        w = w_list[ipatch]
        ipart = 0
        for x_grid in xaxis:
            for y_grid in yaxis:
                dens = density_func(x_grid, y_grid)
                if dens > dens_min:
                    x[ipart:ipart+ppc] = np.random.uniform(-dx/2, dx/2) + x_grid
                    y[ipart:ipart+ppc] = np.random.uniform(-dy/2, dy/2) + y_grid
                    w[ipart:ipart+ppc] = dens / ppc
                    ipart += ppc