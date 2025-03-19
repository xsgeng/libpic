from libpic.patch import Patch
from libpic.particles import ParticlesBase
from numpy import ndarray

def get_npart_to_extend(
    particles_list: list[ParticlesBase],
    patches_list: list[Patch],
    npatches: int, dx: float, dy: float
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Get the number of particles to extend in each patch.
    
    Parameters
    ----------
    particles_list : List[ParticlesBase]
        List of particles of all patches.
    patches_list : List[Patch]
        List of patches
    npatches : int
        Number of patches.
    dx : float
        Cell size in x direction.
    dy : float
        Cell size in y direction.
    
    Returns
    -------
    npart_to_extend : ndarray
        Number of particles to extend in each patch.
    npart_incoming : ndarray
        Number of incoming particles in each patch.
    npart_outgoing : ndarray
        Number of particles outgoing to each boundary in each patch.
    """
    ...
    
def fill_particles_from_boundary(
    particles_list: list[ParticlesBase],
    patches_list: list[Patch],
    npart_incoming: ndarray,
    npart_outgoing: ndarray,
    npatches: int, dx: float, dy: float,
    attrs: list[str]
) -> None:
    """
    Fill the particles from the boundary.
    
    Parameters
    ----------
    particles_list : List[ParticlesBase]
        List of particles of all patches.
    patches_list : List[Patch]
        List of patches.
    npart_incoming : ndarray
        Number of incoming particles in each patch.
    npart_outgoing : ndarray
        Number of particles outgoing to each boundary in each patch.
    npatches : int
        Number of patches.
    dx : float
        Cell size in x direction.
    dy : float
        Cell size in y direction.
    attrs : List[str]
        List of attributes to be synced of the particles.
    """
    ...