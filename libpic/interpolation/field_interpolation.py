import numpy as np
from numba import njit, prange, typed, types
from scipy.constants import c, e, epsilon_0, mu_0

from ..patch import Patches

from .cpu import interpolation_patches_2d


class FieldInterpolation:
    """
    Field interpolation class.

    Holds E, B fields and particle positions of all patches.

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
        self.ex_part_list = []
        self.ey_part_list = []
        self.ez_part_list = []
        self.bx_part_list = []
        self.by_part_list = []
        self.bz_part_list = []
        self.is_dead_list = []

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

        for ispec, s in enumerate(self.patches.species):
            self.x_list.append(typed.List(
                [p.particles[ispec].x for p in self.patches]))
            self.ex_part_list.append(typed.List(
                [p.particles[ispec].ex_part for p in self.patches]))
            self.ey_part_list.append(typed.List(
                [p.particles[ispec].ey_part for p in self.patches]))
            self.ez_part_list.append(typed.List(
                [p.particles[ispec].ez_part for p in self.patches]))
            self.bx_part_list.append(typed.List(
                [p.particles[ispec].bx_part for p in self.patches]))
            self.by_part_list.append(typed.List(
                [p.particles[ispec].by_part for p in self.patches]))
            self.bz_part_list.append(typed.List(
                [p.particles[ispec].bz_part for p in self.patches]))

            self.is_dead_list.append(typed.List(
                [p.particles[ispec].is_dead for p in self.patches]))

    def update_particle_lists(self, ipatch: int, ispec: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        """
        particles = self.patches[ipatch].particles[ispec]

        self.x_list[ispec][ipatch] = particles.x
        self.ex_part_list[ispec][ipatch] = particles.ex_part
        self.ey_part_list[ispec][ipatch] = particles.ey_part
        self.ez_part_list[ispec][ipatch] = particles.ez_part
        self.bx_part_list[ispec][ipatch] = particles.bx_part
        self.by_part_list[ispec][ipatch] = particles.by_part
        self.bz_part_list[ispec][ipatch] = particles.bz_part
        self.is_dead_list[ispec][ipatch] = particles.is_dead

    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.ex_list = typed.List([p.fields.ex for p in self.patches])
        self.ey_list = typed.List([p.fields.ey for p in self.patches])
        self.ez_list = typed.List([p.fields.ez for p in self.patches])
        self.bx_list = typed.List([p.fields.bx for p in self.patches])
        self.by_list = typed.List([p.fields.by for p in self.patches])
        self.bz_list = typed.List([p.fields.bz for p in self.patches])

        self.x0s = np.array([p.x0 for p in self.patches])

    def update_patches(self) -> None:
        """
        Called when the arangement of patches changes.
        """
        self.generate_field_lists()
        self.generate_particle_lists()
        self.npatches = self.patches.npatches
        # raise NotImplementedError

    def __call__(self, ispec: int) -> None:
        """
        Call field interpolation.

        Parameters
        ----------
        ispec : int
            Species index.
        """
        raise NotImplementedError


class FieldInterpolation2D(FieldInterpolation):
    def __init__(self, patches: Patches) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy

    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        for ispec, s in enumerate(self.patches.species):
            self.y_list.append(typed.List(
                [p.particles[ispec].y for p in self.patches]))

    def update_particle_lists(self, ipatch: int, ispec: int):
        super().update_particle_lists(ipatch, ispec)
        self.y_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].y

    def generate_field_lists(self) -> None:
        super().generate_field_lists()
        self.y0s = np.array([p.y0 for p in self.patches])

    def __call__(self, ispec: int) -> None:
        interpolation_patches_2d(
            self.x_list[ispec], self.y_list[ispec],
            self.ex_part_list[ispec], self.ey_part_list[ispec], self.ez_part_list[ispec],
            self.bx_part_list[ispec], self.by_part_list[ispec], self.bz_part_list[ispec],
            self.ex_list, self.ey_list, self.ez_list,
            self.bx_list, self.by_list, self.bz_list,
            self.x0s, self.y0s,
            self.npatches,
            self.dx, self.dy,
            self.is_dead_list[ispec],
        )
