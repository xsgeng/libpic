import numpy as np
from scipy.constants import e, m_e


class ParticlesBase:
    attrs: list[str] = ["x", "y", "w", "ux", "uy", "uz", "inv_gamma",
             "ex_part", "ey_part", "ez_part", "bx_part", "by_part", "bz_part"]
    
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
        self.inv_gamma = np.ones(npart)

        self.ex_part = np.zeros(npart)
        self.ey_part = np.zeros(npart)
        self.ez_part = np.zeros(npart)
        self.bx_part = np.zeros(npart)
        self.by_part = np.zeros(npart)
        self.bz_part = np.zeros(npart)

        self.pruned = np.full(npart, False)

    def extend(self, n : int):
        if n <= 0:
            return
        for attr in self.attrs:
            arr = getattr(self, attr)
            arr = np.append(arr, np.full(n, np.nan))
            setattr(self, attr, arr)
        self.w[-n:] = 0
        self.pruned = np.append(self.pruned, np.full(n, True))
        self.npart += n

    def prune(self):
        for attr in self.attrs:
            setattr(self, attr, getattr(self, attr)[~self.pruned])
        self.pruned = self.pruned[~self.pruned]
        self.npart = len(self.pruned)


class QEDParticles(ParticlesBase):
    def __init__(self) -> None:
        self.attrs += ["chi"]
        super().__init__()


class SpinParticles(ParticlesBase):
    def __init__(self) -> None:
        self.attrs += ["sx", "sy", "sz"]
        super().__init__()


class SpinQEDParticles(SpinParticles, QEDParticles):
    ...