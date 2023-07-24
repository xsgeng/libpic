import inspect
import numpy as np
from numba import jit, njit
from scipy.constants import e, m_e

class Particles:
    attrs = ["x", "y", "x_old", "y_old", "w", "ux", "uy", "uz", "inv_gamma",
             "ex_part", "ey_part", "ez_part", "bx_part", "by_part", "bz_part"]

    def __init__(self, species) -> None:
        self.species = species
        self.q : float = species.charge * e
        self.m : float = species.mass * m_e
    
    @property
    def name(self) -> str:
        return self.species.name

    def initialize(
        self, 
        npart : int,
    ) -> None:
        assert npart >= 0
        self.npart = npart

        self.x = np.zeros(npart)
        self.y = np.zeros(npart)
        self.x_old = np.zeros(npart)
        self.y_old = np.zeros(npart)
        self.w = np.zeros(npart)
        self.ux = np.zeros(npart)
        self.uy = np.zeros(npart)
        self.uz = np.zeros(npart)
        self.inv_gamma = np.ones(npart)

        self.ex_part = np.zeros(npart)
        self.ey_part = np.zeros(npart)
        self.ez_part = np.zeros(npart)
        self.bx_part = np.zeros(npart)
        self.by_part = np.zeros(npart)
        self.bz_part = np.zeros(npart)

        self.pruned = np.full(npart, False)