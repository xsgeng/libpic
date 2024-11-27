from typing import List
import numpy as np
from ..particles import ParticlesBase
from ..fields import Fields

def current_deposition_cpu(
    particles_list: List[ParticlesBase],
    fields_list: List[Fields],
    npatches: int,
    dt: float,
    q: float
) -> None: ...
