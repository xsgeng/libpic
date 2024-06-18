import numpy as np
from numba import njit, prange, typed

from libpic.patch import Patches

from .cpu import current_deposition_cpu, CList


class CurrentDeposition:
    """
    Current deposition class.

    Holds J, Rho fields and some particle attributes of all patches.

    """
    def __init__(self, patches: Patches) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        
        """
        self.patches = patches
        self.npatches: int = patches.npatches
        self.dx: float = patches.dx

        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.w_list = []
        self.ux_list = []
        self.uy_list = []
        self.uz_list = []
        self.inv_gamma_list = []
        self.pruned_list = []

        self.q = []

        self.generate_field_lists()
        self.generate_particle_lists()

    def generate_particle_lists(self) -> None:
        """
        Add species to the current deposition class.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        for ispec, s in enumerate(self.patches.species):
            self.x_list.append(CList([p.particles[ispec].x for p in self.patches]))
            self.w_list.append(CList([p.particles[ispec].w for p in self.patches]))
            self.ux_list.append(CList([p.particles[ispec].ux for p in self.patches]))
            self.uy_list.append(CList([p.particles[ispec].uy for p in self.patches]))
            self.uz_list.append(CList([p.particles[ispec].uz for p in self.patches]))
            self.inv_gamma_list.append(CList([p.particles[ispec].inv_gamma for p in self.patches]))
            self.pruned_list.append(CList([p.particles[ispec].pruned for p in self.patches]))

            self.q.append(s.q)

    def update_particle_lists(self, ipatch: int, ispec: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        particle : Particles
            Particle object in the patch.
        """
        self.x_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].x
        self.w_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].w
        self.ux_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].ux
        self.uy_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].uy
        self.uz_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].uz
        self.inv_gamma_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].inv_gamma
        self.pruned_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].pruned

    
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.jx_list = CList([p.fields.jx for p in self.patches])
        self.jy_list = CList([p.fields.jy for p in self.patches])
        self.jz_list = CList([p.fields.jz for p in self.patches])
        self.rho_list = CList([p.fields.rho for p in self.patches])

        self.x0s = np.array([p.x0 for p in self.patches])

    def update_patches(self) -> None:
        """
        Called when the arangement of patches changes.
        """
        self.generate_field_lists()
        self.generate_particle_lists()
        self.npatches = self.patches.npatches
        # raise NotImplementedError

    def reset(self) -> None:
        """
        Reset J and Rho to zero.
        """
        for ipatch in range(self.npatches):
            self.jx_list[ipatch].fill(0)
            self.jy_list[ipatch].fill(0)
            self.jz_list[ipatch].fill(0)
            self.rho_list[ipatch].fill(0)
        

    def __call__(self, ispec: int, dt: float) -> None:
        """
        Current deposition.

        Parameters
        ----------
        ispec : int
            Species index.
        dt : float
            Time step.
        """
        raise NotImplementedError


class CurrentDeposition2D(CurrentDeposition):
    def __init__(self, patches: Patches) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy


    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        for ispec, s in enumerate(self.patches.species):
            self.y_list.append(CList([p.particles[ispec].y for p in self.patches]))


    def update_particle_lists(self, ipatch: int, ispec: int):
        super().update_particle_lists(ipatch, ispec)
        self.y_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].y


    def generate_field_lists(self) -> None:
        super().generate_field_lists()
        self.y0s = np.array([p.y0 for p in self.patches])


    def __call__(self, ispec:int, dt: float) -> None:
        current_deposition_cpu(
            self.rho_list,
            self.jx_list, self.jy_list, self.jz_list,
            self.x0s, self.y0s,
            self.x_list[ispec], self.y_list[ispec], 
            self.ux_list[ispec], self.uy_list[ispec], self.uz_list[ispec],
            self.inv_gamma_list[ispec],
            self.pruned_list[ispec],
            self.npatches,
            self.dx, self.dy, dt, self.w_list[ispec], self.q[ispec],
        )

class CurrentDeposition3D(CurrentDeposition):
    ...