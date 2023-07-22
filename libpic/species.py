from pydantic import BaseModel, computed_field
from functools import cached_property
from typing import Literal, Callable
from scipy.constants import e, m_e, m_p
from .particles import Particles


class Species(BaseModel):
    name: str
    charge: int = 1
    mass: int = 1
        
    density: Callable = None
    density_min: float = 0
    ppc: int = 0
        
    momentum: tuple[Callable, Callable, Callable] = [None, None, None]
    radiation: Literal["none", "LL", "photons"] = "none"

    @computed_field
    @cached_property
    def q(self) -> float:
        return self.charge * e

    @computed_field
    @cached_property
    def m(self) -> float:
        return self.mass * m_e

    def create_particles(self) -> Particles:
        """ 
        Create Particles from the species.

        Particles class holds the particle data.

        Called by patch. 
        
        Then particles are created within the patch.
        """
        return Particles(self)


class Electron(Species):
    def __init__(self, name='electron', **kwargs) -> None:
        """ shortcut for electron with charge -1 and mass 1 """
        super().__init__(name=name, charge=-1, mass=1, **kwargs)
        

class Positron(Species):
    def __init__(self, name='positron', **kwargs) -> None:
        """ shortcut for positron with charge 1 and mass 1 """
        super().__init__(name=name, charge=1, mass=1, **kwargs)
        
class Proton(Species):
    def __init__(self, name='proton', **kwargs) -> None:
        """ shortcut for proton with charge 1 and mass 1836 """
        super().__init__(name=name, charge=1, mass=int(m_p/m_e), **kwargs)  
        
class Photon(Species):
    def __init__(self, name='photon', **kwargs) -> None:
        """ shortcut for photon with charge 0 and mass 0 """
        super().__init__(name=name, charge=0, mass=0, **kwargs)
