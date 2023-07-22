from numba import float64, typed, types, njit, prange, set_num_threads
from numba.extending import as_numba_type
import numpy as np

from time import perf_counter_ns

from .maxwell_2d import update_bfield_2d, update_efield_2d
from .fields import Fields2D
from .particles import Particles
from .species import Species
from .pusher import boris_cpu
from .deposition_2d import current_deposit_2d

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

        self.particles : list[Particles] = []
    
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

    def add_particles(self, particles: Particles):
        self.particles.append(particles)

class Patches2D:
    """ 
    A container for patches of the fields and particles. 
    The patches will be created by the main class.
    """
    def __init__(self) -> None:
        self.npatches = 0
        self.indexs : list[int] = []
        self.patches : list[Patch2D] = []
        self.species : list[Species] = []
    
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
        """ 
        Gather all the properties in each patch to the numba list.

        Numba jitted functions cannot accept list of objects as input.
        For parallel execution, it is the only option now.
        
        Not needed for GPU kernels since the kernel calls are asynchronous.

        this update is expensive. 
        """

        lists = {}
        for attr in Fields2D.attrs():
            lists[attr] = typed.List([getattr(p.fields, attr) for p in self.patches])

        lists["xaxis"] = typed.List([p.xaxis for p in self.patches])
        lists["yaxis"] = typed.List([p.yaxis for p in self.patches])

        lists["xmin_neighbor_index"] = typed.List([p.xmin_neighbor_index for p in self.patches])
        lists["xmax_neighbor_index"] = typed.List([p.xmax_neighbor_index for p in self.patches])
        lists["ymin_neighbor_index"] = typed.List([p.ymin_neighbor_index for p in self.patches])
        lists["ymax_neighbor_index"] = typed.List([p.ymax_neighbor_index for p in self.patches])

        
        for attr in Particles.attrs():
            lists[attr] = []

            for ispec, s in enumerate(self.species):
                lists[attr].append(typed.List([getattr(p.particles[ispec], attr) for p in self.patches]))

        self.numba_lists = lists


    def sync_guard_fields(self):
        lists = self.lists
        sync_guard_fields(
            lists['ex'], lists['ey'], lists['ez'],
            lists['bx'], lists['by'], lists['bz'],
            lists['jx'], lists['jy'], lists['jz'],
            lists['xmin_neighbor_index'], 
            lists['xmax_neighbor_index'], 
            lists['ymin_neighbor_index'], 
            lists['ymax_neighbor_index'], 
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
        lists = self.numba_lists
        update_efield_patches(
            ex_list = lists['ex'],
            ey_list = lists['ey'],
            ez_list = lists['ez'],
            bx_list = lists['bx'],
            by_list = lists['by'],
            bz_list = lists['bz'],
            jx_list = lists['jx'],
            jy_list = lists['jy'],
            jz_list = lists['jz'],
            npatches = self.npatches, 
            dx = self.dx, 
            dy = self.dy, 
            dt = dt, 
            nx = self.nx,
            ny = self.ny, 
            n_guard = self.n_guard,
        )

    def update_bfield(self, dt):
        lists = self.numba_lists
        update_bfield_patches(
            ex_list = lists['ex'], 
            ey_list = lists['ey'], 
            ez_list = lists['ez'], 
            bx_list = lists['bx'], 
            by_list = lists['by'], 
            bz_list = lists['bz'], 
            npatches = self.npatches, 
            dx = self.dx, 
            dy = self.dy, 
            dt = dt, 
            nx = self.nx,
            ny = self.ny, 
            n_guard = self.n_guard,
        )

    def init_particles(self, species : Species):

        xaxis = typed.List([p.xaxis for p in self.patches])
        yaxis = typed.List([p.yaxis for p in self.patches])
        density_func = njit(species.density)

        num_macro_particles = get_num_macro_particles(
            density_func,
            xaxis, 
            yaxis, 
            self.npatches, 
            species.density_min, 
            species.ppc,
        )

        print(f"Species {species.name} initialized with {sum(num_macro_particles)} macro particles.")

        for ipatch in range(self.npatches):
            particles : Particles = species.create_particles()
            particles.initialize(num_macro_particles[ipatch])
            self[ipatch].add_particles(particles)

        self.species.append(species)

    def fill_particles(self):
        xaxis = typed.List([p.xaxis for p in self.patches])
        yaxis = typed.List([p.yaxis for p in self.patches])
        for ispec, s in enumerate(self.species):
            print(f"Creating Species {s.name}.")
            x_list = typed.List([p.particles[ispec].x for p in self.patches])
            y_list = typed.List([p.particles[ispec].y for p in self.patches])
            w_list = typed.List([p.particles[ispec].w for p in self.patches])
            density_func = njit(s.density)
            fill_particles(
                density_func,
                xaxis, 
                yaxis, 
                self.npatches, 
                s.density_min, 
                s.ppc,
                x_list,y_list,w_list
            )

            for p in self:
                p.particles[ispec].x_old[:] = p.particles[ispec].x[:]
                p.particles[ispec].y_old[:] = p.particles[ispec].y[:]
    
    def push_particles(self, dt):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Pushing Species {s.name}.")
            boris_push(
                lists['ux'][ispec], lists['uy'][ispec], lists['uz'][ispec], lists['inv_gamma'][ispec],
                lists['ex'][ispec], lists['ey'][ispec], lists['ez'][ispec],
                lists['bx'][ispec], lists['by'][ispec], lists['bz'][ispec],
                self.npatches, s.q, lists['npart'][ispec], lists['pruned'][ispec], dt
            )
    
    def current_deposition(self, dt):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Deposition of current for Species {s.name}.")
            current_deposition(
                lists['rho'], lists['jx'], lists['jy'], lists['jz'],
                lists['xaxis'], lists['yaxis'],
                lists['x'][ispec], lists['y'][ispec], lists['uz'][ispec],
                lists['inv_gamma'][ispec], lists['x_old'][ispec], lists['y_old'][ispec],
                lists['pruned'][ispec], lists['npart'][ispec], 
                self.npatches, self.dx, self.dy, dt, lists['w'][ispec], s.q,
            )

            

""" Parallel functions for patches """

@njit(parallel=True)
def update_efield_patches(
    ex_list, ey_list, ez_list, 
    bx_list, by_list, bz_list, 
    jx_list, jy_list, jz_list, 
    npatches,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]

        update_efield_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, dt, nx, ny, n_guard)

@njit(parallel=True)
def update_bfield_patches(
    ex_list, ey_list, ez_list, 
    bx_list, by_list, bz_list, 
    npatches,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

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

@njit(parallel=True)
def boris_push(
    ux_list, uy_list, uz_list, inv_gamma_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches, q, npart_list, pruned_list, dt
) -> None:
    for ipatch in prange(npatches):
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]
        
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        pruned = pruned_list[ipatch]
        npart = npart_list[ipatch]
        for ipart in range(npart):
            if not pruned[ipart]:
                ux[ipart], uy[ipart], uz[ipart], inv_gamma[ipart] = boris_cpu(
                    ux[ipart], uy[ipart], uz[ipart], 
                    ex[ipart], ey[ipart], ez[ipart], 
                    bx[ipart], by[ipart], bz[ipart], 
                    q, dt
                )

@njit(parallel=True)
def current_deposition(
    rho_list, 
    jx_list, jy_list, jz_list,
    xaxis_list, yaxis_list,
    x_list, y_list, uz_list, 
    inv_gamma_list, 
    x_old_list, y_old_list, 
    pruned_list, npart_list, 
    npatches,
    dx, dy, dt, w_list, q,
) -> None:
    for ipatch in prange(npatches):
        rho = rho_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]
        x0 = xaxis_list[ipatch][0]
        y0 = yaxis_list[ipatch][0]
        x = x_list[ipatch]
        y = y_list[ipatch]
        uz = uz_list[ipatch]
        w = w_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]
        x_old = x_old_list[ipatch]
        y_old = y_old_list[ipatch]
        pruned = pruned_list[ipatch]
        npart = npart_list[ipatch]
        current_deposit_2d(rho, jx, jy, jz, x, y, uz, inv_gamma, x_old, y_old, pruned, npart, dx, dy, x0, y0, dt, w, q)