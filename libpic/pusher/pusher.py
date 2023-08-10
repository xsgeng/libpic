from typing import Any

import numpy as np
from numba import njit, prange, typed, types
from scipy.constants import c, e, epsilon_0, mu_0

from libpic.patch import Patches

from .cpu import boris_push_patches, push_position_patches_2d


class PusherBase:
    
    def __init__(self, patches: Patches) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches: int = patches.npatches

        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.ux_list = []
        self.uy_list = []
        self.uz_list = []
        self.inv_gamma_list = []
        self.pruned_list = []

        self.q: list[float] = []
        self.m: list[float] = []


    def generate_particle_lists(self) -> None:
        """
        Generate typed.List of particle data of all species in all patches.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        for ispec, s in enumerate(self.patches.species):
            self.x_list.append(typed.List([p.particles[ispec].x for p in self.patches]))
            if self.dimension == 2:
                self.y_list.append(typed.List([p.particles[ispec].y for p in self.patches]))
            if self.dimension == 3:
                self.z_list.append(typed.List([p.particles[ispec].z for p in self.patches]))
            self.ux_list.append(typed.List([p.particles[ispec].ux for p in self.patches]))
            self.uy_list.append(typed.List([p.particles[ispec].uy for p in self.patches]))
            self.uz_list.append(typed.List([p.particles[ispec].uz for p in self.patches]))
            self.inv_gamma_list.append(typed.List([p.particles[ispec].inv_gamma for p in self.patches]))

            self.pruned_list.append(typed.List([p.particles[ispec].pruned for p in self.patches]))

            self.q.append(s.q)
            self.m.append(s.m)


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
        self.x_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].x
        if self.dimension == 2:
            self.y_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].y
        if self.dimension == 3:
            self.z_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].z
        self.ex_part_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].ex_part
        self.ey_part_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].ey_part
        self.ez_part_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].ez_part
        self.bx_part_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].bx_part
        self.by_part_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].by_part
        self.bz_part_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].bz_part
        self.pruned_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].pruned


    def push_position(self, ispec: int, dt: float):
        if self.dimension == 2:
            push_position_patches_2d(
                self.x_list[ispec], self.y_list[ispec],
                self.ux_list[ispec], self.uy_list[ispec], self.inv_gamma_list[ispec],
                self.pruned_list[ispec], 
                self.npatches, dt,
            )


    def __call__(self, ispec: int, dt: float) -> None:
        raise NotImplementedError
    

class BorisPusher(PusherBase):
    def __call__(self, ispec: int, dt: float) -> None:
        boris_push_patches(
            self.ux_list[ispec], self.uy_list[ispec], self.uz_list[ispec], self.inv_gamma_list[ispec],
            self.ex_list[ispec], self.ey_list[ispec], self.ez_list[ispec],
            self.bx_list[ispec], self.by_list[ispec], self.bz_list[ispec],
            self.pruned_list[ispec],
            self.npatches, self.q[ispec], self.m[ispec], dt
        )