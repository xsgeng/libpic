import numpy as np

class Particles:
    def __init__(self, species) -> None:
        self.species = species
    
    def initialize(self, npart) -> None:
        self.npart = npart

        self.x = np.zeros(npart)
        self.y = np.zeros(npart)
        self.z = np.zeros(npart)
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