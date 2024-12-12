from time import perf_counter_ns

import numpy as np
from numba import njit, typed

from ..boundary.cpml import PML, PMLX, PMLY
from ..boundary.particles import (fill_particles_from_boundary,
                                       get_npart_to_extend,
                                       mark_out_of_bound_as_dead)
from ..fields import Fields, Fields2D
from ..particles import ParticlesBase
from ..patch.cpu import (fill_particles, get_num_macro_particles,
                              sync_currents, sync_guard_fields)
from ..species import Species


class Patch:
    rank: int
    index: int
    ipatch_x: int
    ipatch_y: int
    x0: float
    y0: float

    nx: int
    ny: int
    dx: float
    dy: float

    xaxis: np.ndarray
    yaxis: np.ndarray

    # neighbors
    xmin_neighbor_index: int
    xmax_neighbor_index: int
    ymin_neighbor_index: int
    ymax_neighbor_index: int
    zmin_neighbor_index: int
    zmax_neighbor_index: int

    # MPI neighbors
    xmin_neighbor_rank: int
    xmax_neighbor_rank: int
    ymin_neighbor_rank: int
    ymax_neighbor_rank: int
    zmin_neighbor_rank: int
    zmax_neighbor_rank: int

    fields: Fields
    def __init__(self) -> None:
        # PML boundaries
        self.pml_boundary: list[PML] = []

        self.particles : list[ParticlesBase] = []

    @property
    def xmin(self):
        return self.x0
    @property
    def xmax(self):
        return self.x0 + (self.nx-1) * self.dx
    @property
    def ymin(self):
        return self.y0
    @property
    def ymax(self):
        return self.y0 + (self.ny-1) * self.dy

    def add_particles(self, particles: ParticlesBase) -> None:
        self.particles.append(particles)

    def set_fields(self, fields: Fields) -> None:
        self.fields = fields

    def add_pml_boundary(self, pml: PML) -> None:
        raise NotImplementedError

class Patch2D(Patch):
    def __init__(
        self,
        rank: int,
        index: int,
        ipatch_x: int,
        ipatch_y: int,
        x0 : float, 
        y0 : float,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
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
        super().__init__()
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

        self.xaxis = np.arange(self.nx) * self.dx + x0
        self.yaxis = np.arange(self.ny) * self.dy + y0

        # neighbors
        self.xmin_neighbor_index: int = -1
        self.xmax_neighbor_index: int = -1
        self.ymin_neighbor_index: int = -1
        self.ymax_neighbor_index: int = -1

        # MPI neighbors
        self.xmin_neighbor_rank: int = -1
        self.xmax_neighbor_rank: int = -1
        self.ymin_neighbor_rank: int = -1
        self.ymax_neighbor_rank: int = -1

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

    def add_pml_boundary(self, pml: PML) -> None:
        assert (self.nx >= pml.thickness) and (self.ny >= pml.thickness)
        assert len(self.pml_boundary) < 2, "cannot assign more than 2 PML boundaries to one patch, \
                                             try increasing number of patches."
        if len(self.pml_boundary) == 1:
            assert isinstance(self.pml_boundary[0], PMLX) ^ isinstance(pml, PMLX)
            # assert isinstance(self.pml_boundary[0], PMLY) ^ isinstance(pml, PMLY)
        self.pml_boundary.append(pml)



class Patches:
    """ 
    A container for patches of the fields and particles. 

    The class handles synchronization of fields and particles between patches.
    """
    def __init__(self, dimension: int) -> None:
        assert (dimension == 1) or (dimension == 2) or (dimension == 3)
        self.dimension: int = dimension

        self.npatches: int = 0
        self.indexs : list[int] = []
        self.patches : list[Patch] = []
        self.species : list[Species] = []
    
    def __getitem__(self, i: int) -> Patch:
        return self.patches[i]

    def __len__(self) -> int:
        return self.npatches

    def __iter__(self):
        return iter(self.patches)

    def append(self, patch: Patch):
        if self.patches:
            assert self.patches[-1].index == patch.index - 1
        self.patches.append(patch)
        self.indexs.append(patch.index)
        self.npatches += 1

    def prepend(self, patch: Patch):
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
        for attr in Fields2D.attrs:
            lists[attr] = typed.List([getattr(p.fields, attr) for p in self.patches])

        lists["xaxis"] = typed.List([p.xaxis for p in self.patches])
        lists["yaxis"] = typed.List([p.yaxis for p in self.patches])

        lists["xmin_neighbor_index"] = np.array([p.xmin_neighbor_index for p in self.patches])
        lists["xmax_neighbor_index"] = np.array([p.xmax_neighbor_index for p in self.patches])
        lists["ymin_neighbor_index"] = np.array([p.ymin_neighbor_index for p in self.patches])
        lists["ymax_neighbor_index"] = np.array([p.ymax_neighbor_index for p in self.patches])
        self.grid_lists = lists

        particle_lists = []
        for ispec, s in enumerate(self.species):
            particle_lists.append({})
            lists = particle_lists[ispec]
            lists["npart"] = typed.List([p.particles[ispec].npart for p in self.patches])
            lists["is_dead"] = typed.List([p.particles[ispec].is_dead for p in self.patches])

            for attr in self[0].particles[ispec].attrs:
                lists[attr] = typed.List([getattr(p.particles[ispec], attr) for p in self.patches])

        self.particle_lists = particle_lists


    def update_particle_lists(self, ipatch):
        plists = self.particle_lists
        patch = self[ipatch]
        for ispec, s in enumerate(self.species):
            plists[ispec]["npart"][ipatch] = patch.particles[ispec].npart
            plists[ispec]["is_dead"][ipatch] = patch.particles[ispec].is_dead

            for attr in patch.particles[ispec].attrs:
                plists[ispec][attr][ipatch] = getattr(patch.particles[ispec], attr)

    def sync_guard_fields(self):
        lists = self.grid_lists
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


    def sync_currents(self):
        lists = self.grid_lists
        sync_currents(
            lists['jx'], lists['jy'], lists['jz'], lists["rho"],
            lists['xmin_neighbor_index'], 
            lists['xmax_neighbor_index'], 
            lists['ymin_neighbor_index'], 
            lists['ymax_neighbor_index'], 
            self.npatches, 
            self.nx,
            self.ny,
            self.n_guard,
        )

    def sync_particles(self) -> None:
        lists = self.grid_lists
        plists = self.particle_lists
        for ispec, s in enumerate(self.species):

            npart_to_extend, npart_incoming, npart_outgoing = get_npart_to_extend(
                plists[ispec]["x"], plists[ispec]["y"],
                plists[ispec]["npart"], plists[ispec]["is_dead"],
                lists["xaxis"], lists["yaxis"],
                lists["xmin_neighbor_index"], lists["xmax_neighbor_index"], 
                lists["ymin_neighbor_index"], lists["ymax_neighbor_index"],
                self.npatches, self.dx, self.dy,
            )

            # extend the particles in each patch in python mode
            # TODO: extend the particles in each patch in numba mode in parallel
            # typed.List cannot modify the attr of the particle object since
            # the address is modified after being extended.
            for ipatches in range(self.npatches):
                p = self[ipatches].particles[ispec]
                if npart_to_extend[ipatches] > 0:
                    p.extend(npart_to_extend[ipatches])
                    p.extended = True
                    self.update_particle_lists(ipatches)

            fill_particles_from_boundary(
                plists[ispec]["is_dead"],
                lists["xaxis"], lists["yaxis"],
                lists["xmin_neighbor_index"], lists["xmax_neighbor_index"], lists["ymin_neighbor_index"], lists["ymax_neighbor_index"],
                npart_incoming, npart_outgoing,
                self.npatches, self.dx, self.dy,
                *[plists[ispec][attr] for attr in self[ipatches].particles[ispec].attrs],
            )

            mark_out_of_bound_as_dead(
                plists[ispec]["x"], plists[ispec]["y"],
                plists[ispec]["npart"], plists[ispec]["is_dead"],
                lists["xaxis"], lists["yaxis"],
                self.npatches, self.dx, self.dy,
            )

        
    @property
    def nx(self) -> int:
        return self[0].fields.nx

    @property
    def ny(self) -> int:
        return self[0].fields.ny

    @property
    def dx(self) -> float:
        return self[0].fields.dx

    @property
    def dy(self) -> float:
        return self[0].fields.dy


    @property
    def n_guard(self):
        return self[0].fields.n_guard


    def add_species(self, species : Species):

        print(f"Initializing Species {species.name}...", end=" ")
        tic = perf_counter_ns()
        xaxis = typed.List([p.xaxis for p in self.patches])
        yaxis = typed.List([p.yaxis for p in self.patches])

        if species.density is not None:

            num_macro_particles = get_num_macro_particles(
                species.density_jit,
                xaxis, 
                yaxis, 
                self.npatches, 
                species.density_min, 
                species.ppc,
            )
        else:
            num_macro_particles = np.zeros(self.npatches, dtype='int64')


        for ipatch in range(self.npatches):
            particles : ParticlesBase = species.create_particles(ipatch=ipatch)
            particles.initialize(num_macro_particles[ipatch])
            self[ipatch].add_particles(particles)

        self.species.append(species)
        
        num_macro_particles_sum = sum(num_macro_particles)
        print(f"{(perf_counter_ns() - tic)/1e6} ms.")
        print(f"Species {species.name} initialized with {sum(num_macro_particles)} macro particles.")
        return num_macro_particles_sum

    def fill_particles(self):
        xaxis = typed.List([p.xaxis for p in self.patches])
        yaxis = typed.List([p.yaxis for p in self.patches])
        for ispec, s in enumerate(self.species):
            print(f"Creating Species {s.name}...", end=" ")
            tic = perf_counter_ns()
            if s.density is not None:
                x_list = typed.List([p.particles[ispec].x for p in self.patches])
                y_list = typed.List([p.particles[ispec].y for p in self.patches])
                w_list = typed.List([p.particles[ispec].w for p in self.patches])
                fill_particles(
                    s.density_jit,
                    xaxis, 
                    yaxis, 
                    self.npatches, 
                    s.density_min, 
                    s.ppc,
                    x_list,y_list,w_list
                )
    
            print(f"{(perf_counter_ns() - tic)/1e6} ms.")