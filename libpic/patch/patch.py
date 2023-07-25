from numba import float64, typed, types, njit, set_num_threads
from numba.extending import as_numba_type
import numpy as np

from time import perf_counter_ns

from libpic.fields import Fields2D
from libpic.particles import Particles
from libpic.patch.cpu import (
    boris_push, current_deposition, fill_particles, 
    fill_particles_from_boundary, get_npart_to_extend, 
    get_num_macro_particles, mark_out_of_bound_as_pruned, 
    push_position, sync_guard_fields, update_bfield_patches, 
    update_efield_patches, interpolation
)
from libpic.species import Species

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

        self.numba_lists = lists

    def update_particle_lists(self, ipatch):
        lists = self.numba_lists
        patch = self[ipatch]
        for ispec, s in enumerate(self.species):
            lists["npart"][ispec][ipatch] = patch.particles[ispec].npart
            lists["pruned"][ispec][ipatch] = patch.particles[ispec].pruned

            for attr in Particles.attrs:
                lists[attr][ispec][ipatch] = getattr(patch.particles[ispec], attr)


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
            mark_out_of_bound_as_pruned(
                lists["x"][ispec], lists["y"][ispec],
                lists["npart"][ispec], lists["pruned"][ispec],
                lists["xaxis"], lists["yaxis"],
                self.npatches, self.dx, self.dy,
            )

            npart_to_extend, npart_new_list = get_npart_to_extend(
                lists["x"][ispec], lists["y"][ispec],
                lists["npart"][ispec], lists["pruned"][ispec],
                lists["xaxis"], lists["yaxis"],
                lists["xmin_neighbor_index"], lists["xmax_neighbor_index"], lists["ymin_neighbor_index"], lists["ymax_neighbor_index"],
                self.npatches, self.dx, self.dy,
            )

            for ipatches in range(self.npatches):
                if npart_to_extend[ipatches] > 0:
                    self.patches[ipatches].particles[ispec].extend(npart_to_extend[ipatches])
                    self.update_particle_lists(ipatches)
            fill_particles_from_boundary(
                lists["npart"][ispec], lists["pruned"][ispec],
                lists["xaxis"], lists["yaxis"],
                lists["xmin_neighbor_index"], lists["xmax_neighbor_index"], lists["ymin_neighbor_index"], lists["ymax_neighbor_index"],
                npart_new_list,
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


    def momentum_push(self, dt):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Pushing Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            boris_push(
                lists['ux'][ispec], lists['uy'][ispec], lists['uz'][ispec], lists['inv_gamma'][ispec],
                lists['ex_part'][ispec], lists['ey_part'][ispec], lists['ez_part'][ispec],
                lists['bx_part'][ispec], lists['by_part'][ispec], lists['bz_part'][ispec],
                self.npatches, s.q, lists['npart'][ispec], lists['pruned'][ispec], dt
            )
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")
    
    def position_push(self, dt):
        lists = self.numba_lists
        for ispec, s in enumerate(self.species):
            print(f"Pushing Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            push_position(
                lists['x'][ispec], lists['y'][ispec], 
                lists['ux'][ispec], lists['uy'][ispec], lists['inv_gamma'][ispec],
                self.npatches, lists['pruned'][ispec], dt
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
                lists['x'][ispec], lists['y'][ispec], 
                lists['ux'][ispec], lists['uy'][ispec], lists['uz'][ispec], lists['inv_gamma'][ispec], 
                lists['pruned'][ispec], 
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

            







