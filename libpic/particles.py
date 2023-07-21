import inspect
import numpy as np
from numba import jit, njit
from scipy.constants import e, m_e

class Particles:
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
        self.w = np.zeros(npart)
        self.ux = np.zeros(npart)
        self.uy = np.zeros(npart)
        self.uz = np.zeros(npart)

        self.ex = np.zeros(npart)
        self.ey = np.zeros(npart)
        self.ez = np.zeros(npart)
        self.bx = np.zeros(npart)
        self.by = np.zeros(npart)
        self.bz = np.zeros(npart)

        self.pruned = np.full(npart, False)