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
from .interpolation_2d import interpolation_2d

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

        
        lists["npart"] = []
        lists["pruned"] = []
        for ispec, s in enumerate(self.species):
            lists["npart"].append(typed.List([p.particles[ispec].npart for p in self.patches]))
            lists["pruned"].append(typed.List([p.particles[ispec].pruned for p in self.patches]))

        for attr in Particles.attrs:
            lists[attr] = []
            for ispec, s in enumerate(self.species):
                lists[attr].append(typed.List([getattr(p.particles[ispec], attr) for p in self.patches]))
                lists["npart"].append(typed.List([getattr(p.particles[ispec], attr) for p in self.patches]))
        self.numba_lists = lists


    def sync_guard_fields(self):
        lists = self.lists
        print(f"Synching guard fields...", end=" ")
        tic = perf_counter_ns()
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
        print(f"{(perf_counter_ns() - tic)/1e6} ms.")

    def sync_particles(self):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Synching Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            sync_particles(
                lists["npart"][ispec], lists["pruned"][ispec],
                lists["xaxis"], lists["yaxis"],
                lists["xmin_neighbor_index"], lists["xmax_neighbor_index"], lists["ymin_neighbor_index"], lists["ymax_neighbor_index"],
                self.npatches, self.dx, self.dy,
                *[lists[attr][ispec] for attr in Particles.attrs]
            )
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")

        
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
        print(f"Updating B field...", end=" ")
        tic = perf_counter_ns()
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
        print(f"{(perf_counter_ns() - tic)/1e6} ms.")

    def update_bfield(self, dt):
        lists = self.numba_lists
        print(f"Updating B field...", end=" ")
        tic = perf_counter_ns()
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
        print(f"{(perf_counter_ns() - tic)/1e6} ms.")

    def init_particles(self, species : Species):

        print(f"Initializing Species {species.name}...", end=" ")
        tic = perf_counter_ns()
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


        for ipatch in range(self.npatches):
            particles : Particles = species.create_particles()
            particles.initialize(num_macro_particles[ipatch])
            self[ipatch].add_particles(particles)

        self.species.append(species)
        print(f"{(perf_counter_ns() - tic)/1e6} ms.")
        print(f"Species {species.name} initialized with {sum(num_macro_particles)} macro particles.")

    def fill_particles(self):
        xaxis = typed.List([p.xaxis for p in self.patches])
        yaxis = typed.List([p.yaxis for p in self.patches])
        for ispec, s in enumerate(self.species):
            print(f"Creating Species {s.name}...", end=" ")
            tic = perf_counter_ns()
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
    
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")


    def push_particles(self, dt):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Pushing Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            boris_push(
                lists['ux'][ispec], lists['uy'][ispec], lists['uz'][ispec], lists['inv_gamma'][ispec],
                lists['ex'][ispec], lists['ey'][ispec], lists['ez'][ispec],
                lists['bx'][ispec], lists['by'][ispec], lists['bz'][ispec],
                self.npatches, s.q, lists['npart'][ispec], lists['pruned'][ispec], dt
            )
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")
    
    def current_deposition(self, dt):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Deposition of current for Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            current_deposition(
                lists['rho'], lists['jx'], lists['jy'], lists['jz'],
                lists['xaxis'], lists['yaxis'],
                lists['x'][ispec], lists['y'][ispec], lists['uz'][ispec],
                lists['inv_gamma'][ispec], lists['x_old'][ispec], lists['y_old'][ispec],
                lists['pruned'][ispec], lists['npart'][ispec], 
                self.npatches, self.dx, self.dy, dt, lists['w'][ispec], s.q,
            )
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")

    def interpolation(self):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Interpolation of current for Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            interpolation(
                lists["x"][ispec], lists["y"][ispec], 
                lists["ex_part"][ispec], lists["ey_part"][ispec], lists["ez_part"][ispec],
                lists["bx_part"][ispec], lists["by_part"][ispec], lists["bz_part"][ispec],
                lists["npart"][ispec], 
                lists["ex"], lists["ey"], lists["ez"],
                lists["bx"], lists["by"], lists["bz"],
                lists["xaxis"], lists["yaxis"],
                self.npatches, self.dx, self.dy, lists["pruned"][ispec],
            )
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")

            

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


@njit
def count_new_particles(
    x_list, y_list, npart_list,
    xmin_index, xmax_index, ymin_index, ymax_index,
    xaxis_list, yaxis_list,
    dx, dy,
):
    npart_out_of_bound = 0
       
    xmax = xaxis_list[xmin_index][-1] + 0.5*dx
    x = x_list[xmin_index]
    for i in range(npart_list[xmin_index]):
        if x[i] > xmax:
            npart_out_of_bound += 1

    xmin = xaxis_list[xmax_index][0] - 0.5*dx
    x = x_list[xmax_index]
    for i in range(npart_list[xmax_index]):
        if x[i] < xmin:
            npart_out_of_bound += 1

    ymax = yaxis_list[ymin_index][-1] + 0.5*dy
    y = y_list[ymin_index]
    for i in range(npart_list[ymin_index]):
        if y[i] > ymax:
            npart_out_of_bound += 1

    ymin = yaxis_list[ymax_index][0] - 0.5*dy
    y = y_list[ymax_index]
    for i in range(npart_list[ymax_index]):
        if y[i] < ymin:
            npart_out_of_bound += 1

    return npart_out_of_bound

@njit(inline="always")
def fill_boundary_particles_to_buffer(
    buffer,
    npart_list, xaxis_list, yaxis_list,
    xmin_index, xmax_index, ymin_index, ymax_index,
    dx, dy,
    attrs_list,
):
    npart_new = buffer.shape[0]
    ibuff = 0
    # on xmin boundary
    x_on_xmin = attrs_list[0][xmin_index]
    xmax = xaxis_list[xmin_index][-1] + 0.5*dx
    for ipart in range(npart_list[xmin_index]):
        if ibuff >= npart_new:
            break
        if x_on_xmin[ipart] > xmax:
            for iattr, attr in enumerate(attrs_list):
                buffer[ibuff, iattr] = attr[xmin_index][ipart]
            ibuff += 1

    # on xmax boundary
    x_on_xmax = attrs_list[0][xmax_index]
    xmin = xaxis_list[xmax_index][-1] - 0.5*dx
    for ipart in range(npart_list[xmax_index]):
        if ibuff >= npart_new:
            break
        if x_on_xmax[ipart] < xmin:
            for iattr, attr in enumerate(attrs_list):
                buffer[ibuff, iattr] = attr[xmax_index][ipart]
            ibuff += 1
    
    # on ymin boundary
    y_on_ymin = attrs_list[1][ymin_index]
    ymax = yaxis_list[ymin_index][-1] + 0.5*dy
    for ipart in range(npart_list[ymin_index]):
        if ibuff >= npart_new:
            break
        if y_on_ymin[ipart] > ymax:
            for iattr, attr in enumerate(attrs_list):
                buffer[ibuff, iattr] = attr[ymin_index][ipart]
            ibuff += 1

    # on ymax boundary
    y_on_ymax = attrs_list[1][ymax_index]
    ymin = yaxis_list[ymax_index][ 0] - 0.5*dy
    for ipart in range(npart_list[ymax_index]):
        if ibuff >= npart_new:
            break
        if y_on_ymax[ipart] < ymin:
            for iattr, attr in enumerate(attrs_list):
                buffer[ibuff, iattr] = attr[ymax_index][ipart]
            ibuff += 1
    
    assert ibuff == npart_new


@njit(parallel=True, cache=True)
def sync_particles(
    npart_list,
    pruned_list,
    xaxis_list,
    yaxis_list,
    xmin_index_list, 
    xmax_index_list, 
    ymin_index_list, 
    ymax_index_list, 
    npatches, dx, dy,
    *attrs_list,
):
    """ 
    put particles in neighbor patch into pruned particles.
    """
    nattrs = len(attrs_list)

    for ipatches in prange(npatches):
        x = attrs_list[0][ipatches]
        y = attrs_list[1][ipatches]
        npart = npart_list[ipatches]
        pruned = pruned_list[ipatches]
        xaxis = xaxis_list[ipatches]
        yaxis = yaxis_list[ipatches]
        # mark pruned
        for i in range(npart):
            if x[i] > xaxis[-1] + 0.5*dx or x[i] < xaxis[0] - 0.5*dx or y[i] > yaxis[-1] + 0.5*dy or y[i] < yaxis[0] - 0.5*dy:
                pruned[i] = True
    
    for ipatches in prange(npatches):
        x = attrs_list[0][ipatches]
        y = attrs_list[1][ipatches]

        pruned = pruned_list[ipatches]

        xmin_index = xmin_index_list[ipatches]
        xmax_index = xmax_index_list[ipatches]
        ymin_index = ymin_index_list[ipatches]
        ymax_index = ymax_index_list[ipatches]

        xaxis = xaxis_list[ipatches]
        yaxis = yaxis_list[ipatches]
        
        # 0, 1 for x and y
        npart_new = count_new_particles(attrs_list[0], attrs_list[1], npart_list, 
                                        xmin_index, xmax_index, ymin_index, ymax_index, xaxis_list, yaxis_list, dx, dy)
        if npart_new == 0:
            continue
        buffer = np.zeros((npart_new, nattrs))
        fill_boundary_particles_to_buffer(buffer, npart_list, xaxis_list, yaxis_list,
                                          xmin_index, xmax_index, ymin_index, ymax_index, 
                                          dx, dy, attrs_list)
        
        npart_to_extend = npart_new - sum(pruned)
        if npart_to_extend > 0:
            # reserved 25% more space for new particles
            npart_to_extend += int(len(pruned) * 0.25)
            for i in range(nattrs):
                attrs_list[i][ipatches] = np.append(attrs_list[i][ipatches], np.zeros(npart_to_extend))
            pruned = np.append(pruned, np.full(npart_to_extend, True))

        # fill the pruned
        ibuff = 0
        for ipart in range(len(pruned)):
            if ibuff >= npart_new:
                break
            if pruned[ipart]:
                for iattr in range(nattrs):
                    attrs_list[iattr][ipatches][ipart] = buffer[ibuff, iattr]
            pruned[ipart] = False
            ibuff += 1
        

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
        npart = len(pruned)
        current_deposit_2d(rho, jx, jy, jz, x, y, uz, inv_gamma, x_old, y_old, pruned, npart, dx, dy, x0, y0, dt, w, q)

@njit(parallel=True)
def interpolation(
    x_list, y_list, 
    ex_part_list, ey_part_list, ez_part_list, 
    bx_part_list, by_part_list, bz_part_list, 
    npart_list,
    ex_list, ey_list, ez_list, 
    bx_list, by_list, bz_list,
    xaxis_list, yaxis_list,
    npatches,
    dx, dy,
    pruned_list,
) -> None:
    for ipatch in prange(npatches):
        x = x_list[ipatch]
        y = y_list[ipatch]
        ex_part = ex_part_list[ipatch]
        ey_part = ey_part_list[ipatch]
        ez_part = ez_part_list[ipatch]
        bx_part = bx_part_list[ipatch]
        by_part = by_part_list[ipatch]
        bz_part = bz_part_list[ipatch]
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        x0 = xaxis_list[ipatch][0]
        y0 = yaxis_list[ipatch][0]
        x = x_list[ipatch]
        y = y_list[ipatch]
        pruned = pruned_list[ipatch]
        npart = len(pruned)
        interpolation_2d(
            x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
            ex, ey, ez, bx, by, bz,
            dx, dy, x0, y0,
            pruned,
        )