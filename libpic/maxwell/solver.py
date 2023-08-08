
from typing import Any
import numpy as np
from numba import njit, prange, typed, types
from scipy.constants import c, e, epsilon_0, mu_0

from libpic.patch import Patches2D

from ..patch import Patches2D
from .cpu import update_efield_patches_2d, update_bfield_patches_2d


class MaxwellSolver:
    def __init__(self, patches: Patches2D) -> None:
        self.patches = patches
        self.npatches: int = patches.npatches

        self.dx: float = patches.dx
        self.nx: int = patches.nx

        self.n_guard: int = patches.n_guard

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
        self.jx_list = typed.List([p.fields.jx for p in self.patches])
        self.jy_list = typed.List([p.fields.jy for p in self.patches])
        self.jz_list = typed.List([p.fields.jz for p in self.patches])


    def update_efield(self, dt: float) -> Any:
        raise NotImplementedError

    def update_bfield(self, dt: float) -> Any:
        raise NotImplementedError

class MaxwellSolver2d(MaxwellSolver):
    def __init__(self, patches: Patches2D) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy
        self.ny: int = patches.ny

    def update_efield(self, dt: float) -> Any:
        update_efield_patches_2d(
            self.ex_list, self.ey_list, self.ez_list,
            self.bx_list, self.by_list, self.bz_list,
            self.jx_list, self.jy_list, self.jz_list,
            self.npatches,
            self.dx, self.dy, dt,
            self.nx, self.ny, self.n_guard,
        )

        
    def update_bfield(self, dt: float) -> Any:
        update_bfield_patches_2d(
            self.ex_list, self.ey_list, self.ez_list,
            self.bx_list, self.by_list, self.bz_list,
            self.npatches,
            self.dx, self.dy, dt,
            self.nx, self.ny, self.n_guard,
        )


class MaxwellSolver3d(MaxwellSolver):
    ...