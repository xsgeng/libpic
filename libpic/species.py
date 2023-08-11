from pydantic import BaseModel, computed_field
from functools import cached_property
from typing import Literal, Callable
from scipy.constants import e, m_e, m_p
from .particles import ParticlesBase


class Species(BaseModel):
    name: str
    charge: int
    mass: float
        
    density: Callable = None
    density_min: float = 0
    ppc: int = 0
        
    momentum: tuple[Callable, Callable, Callable] = [None, None, None]
    radiation: Literal["LL", "photons"] = None
    photon_name: str = None
    bw_pair_name: list[str] = [None, None]

    @computed_field
    @cached_property
    def q(self) -> float:
        return self.charge * e

    @computed_field
    @cached_property
    def m(self) -> float:
        return self.mass * m_e

    def create_particles(self) -> ParticlesBase:
        """ 
        Create Particles from the species.

        Particles class holds the particle data.

        Called by patch. 
        
        Then particles are created within the patch.
        """
        return ParticlesBase()

class Electron(Species):
    name: str = 'electron'
    charge: int = -1
    mass: float = 1
        

class Positron(Species):
    def __init__(self, name='positron', **kwargs) -> None:
        """ shortcut for positron with charge 1 and mass 1 """
        super().__init__(name=name, charge=1, mass=1, **kwargs)
        
class Proton(Species):
    def __init__(self, name='proton', **kwargs) -> None:
        """ shortcut for proton with charge 1 and mass 1836 """
        super().__init__(name=name, charge=1, mass=m_p/m_e, **kwargs)  
        
class Photon(Species):
    def __init__(self, name='photon', **kwargs) -> None:
        """ shortcut for photon with charge 0 and mass 0 """
        super().__init__(name=name, charge=0, mass=0, **kwargs)
