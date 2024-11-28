import numpy as np
from scipy.constants import e, m_e
from numpy.typing import NDArray

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

    npart: int # length of the particles, including dead

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
        self.extended = True

    def prune(self, extra_buff=0.1):
        n_alive = self.is_alive.sum()
        npart = int(n_alive * (1 + extra_buff))
        if npart >= self.npart:
            return
        sorted_idx = np.argsort(self.is_dead)
        for attr in self.attrs:
            arr: NDArray[np.float64] = getattr(self, attr)
            arr[:] = arr[sorted_idx]
            arr.resize(npart, refcheck=False)

        self.is_dead[:] = self.is_dead[sorted_idx]
        self.is_dead.resize(npart, refcheck=False)
        self.npart = npart
        self.extended = True
        return sorted_idx

    @property
    def is_alive(self) -> np.ndarray:
        return np.logical_not(self.is_dead)


class QEDParticles(ParticlesBase):
    def __init__(self) -> None:
        super().__init__()
        self.attrs += ["chi", "tau", "delta"]

    def initialize(self, npart: int) -> None:
        super().initialize(npart)
        self.event = np.full(npart, False)

    def extend(self, n: int):
        self.event.resize(n + self.npart, refcheck=False)
        self.event[-n:] = False
        super().extend(n)

    def prune(self, extra_buff=0.1):
        sorted_idx = super().prune(extra_buff=extra_buff)
        self.event[:] = self.event[sorted_idx]
        self.event.resize(self.npart, refcheck=False)


class SpinParticles(ParticlesBase):
    def __init__(self) -> None:
        super().__init__()
        self.attrs += ["sx", "sy", "sz"]


class SpinQEDParticles(SpinParticles, QEDParticles):
    ...
