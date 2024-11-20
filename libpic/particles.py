import numpy as np
from scipy.constants import e, m_e


class ParticlesBase:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    inv_gamma: np.ndarray

    ex_part: np.ndarray
    ey_part: np.ndarray
    ez_part: np.ndarray
    bx_part: np.ndarray
    by_part: np.ndarray
    bz_part: np.ndarray

    def __init__(self) -> None:
        self.attrs: list[str] = [
            "x", "y", "w", "ux", "uy", "uz", "inv_gamma",
            "ex_part", "ey_part", "ez_part", "bx_part", "by_part", "bz_part"
        ]
        self.extended: bool = False

    def initialize(
        self,
        npart: int,
    ) -> None:
        assert npart >= 0
        self.npart = npart

        for attr in self.attrs:
            setattr(self, attr, np.zeros(npart))

        self.inv_gamma[:] = 1
        self.is_dead = np.full(npart, False)

    def extend(self, n: int):
        if n <= 0:
            return
        for attr in self.attrs:
            arr: np.ndarray = getattr(self, attr)
            arr.resize(n + self.npart, refcheck=False)
            # new data set to nan
            arr[-n:] = np.nan

        self.w[-n:] = 0

        self.is_dead.resize(n + self.npart, refcheck=False)
        self.is_dead[-n:] = True

        self.npart += n

    def prune(self):
        for attr in self.attrs:
            setattr(self, attr, getattr(self, attr)[~self.is_dead])
        self.is_dead = self.is_dead[np.logical_not(self.is_dead)]
        self.npart = len(self.is_dead)


class QEDParticles(ParticlesBase):
    def __init__(self) -> None:
        super().__init__()
        self.attrs += ["chi", "tau"]

    def initialize(self, npart: int) -> None:
        super().initialize(npart)
        self.event = np.full(npart, False)

    def extend(self, n: int):
        super().extend(n)
        self.event = np.append(self.event, np.full(n, False))

    def prune(self):
        super().prune()
        self.event = self.event[np.logical_not(self.is_dead)]


class SpinParticles(ParticlesBase):
    def __init__(self) -> None:
        super().__init__()
        self.attrs += ["sx", "sy", "sz"]


class SpinQEDParticles(SpinParticles, QEDParticles):
    ...
