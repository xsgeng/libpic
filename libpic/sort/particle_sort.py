import numpy as np
from libpic.patch.patch import Patches
from numba import typed

from .cpu import sort_particles_patches

class ParticleSort2D:
    """
    sort after particle sync: no particles in guard cells
    """
    def __init__(self, patches: Patches, ispec: int) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        patches : Patches
            Patches to be sorted.
        ispec : int
            Species index.
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches: int = patches.npatches
        self.ispec = ispec
        self.attrs = patches[0].particles[ispec].attrs
        
        self.dx: float = patches.dx
        self.dy: float = patches.dy
        self.nx: int = patches.nx
        self.ny: int = patches.ny

        self.n_guard: int = patches.n_guard

        self.generate_particle_lists()
        self.generate_field_lists()

    def generate_particle_lists(self) -> None:
        """
        Generate typed.List of particle data of all species in all patches.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        ispec = self.ispec
        
        self.x_list = typed.List([p.particles[ispec].x for p in self.patches])
        if self.dimension >= 2:
            self.y_list = typed.List([p.particles[ispec].y for p in self.patches])
        if self.dimension == 3:
            self.z_list = typed.List([p.particles[ispec].z for p in self.patches])

        self.attrs_list = []
        for attr in self.attrs:
            self.attrs_list.append(typed.List([getattr(p.particles[ispec], attr) for p in self.patches]))

        self.pruned_list = typed.List([p.particles[ispec].pruned for p in self.patches])
        
        self.particle_cell_indices_list = typed.List([np.full(p.particles[ispec].pruned.size, -1, dtype=int) for p in self.patches])
        self.sorted_indices_list = typed.List([np.full(p.particles[ispec].pruned.size, -1, dtype=int) for p in self.patches])

    def update_particle_lists(self, ipatch: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        """

        particles = self.patches[ipatch].particles[self.ispec]

        self.x_list[ipatch] = particles.x
        if self.dimension >= 2:
            self.y_list[ipatch] = particles.y
        if self.dimension == 3:
            self.z_list[ipatch] = particles.z

        for iattr, attr in enumerate(self.attrs):
            self.attrs_list[iattr][ipatch] = getattr(particles, attr)
                
        self.pruned_list[ipatch] = particles.pruned
        
        self.particle_cell_indices_list[ipatch] = np.full(particles.pruned.size, -1, dtype=int)
        self.sorted_indices_list[ipatch] = np.full(particles.pruned.size, -1, dtype=int)
        
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.grid_cell_count_list = typed.List([np.full((self.nx, self.ny), 0, dtype=int) for _ in range(self.npatches)])
        self.cell_bound_min_list = typed.List([np.full((self.nx, self.ny), -1, dtype=int) for _ in range(self.npatches)])
        self.cell_bound_max_list = typed.List([np.full((self.nx, self.ny), -1, dtype=int) for _ in range(self.npatches)])

        self.x0s = np.array([p.x0 - self.dx/2 for p in self.patches])
        self.y0s = np.array([p.y0 - self.dy/2 for p in self.patches])
        
    def __call__(self) -> None:
        sort_particles_patches(
            self.grid_cell_count_list, self.cell_bound_min_list, self.cell_bound_max_list, self.x0s, self.y0s,
            self.nx, self.ny, self.dx, self.dy, self.particle_cell_indices_list, self.sorted_indices_list, self.x_list, self.y_list, self.pruned_list, *self.attrs_list
        )
