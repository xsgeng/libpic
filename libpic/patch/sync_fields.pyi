from typing import List
import numpy as np
from ...fields import Fields

def sync_currents_2d(
    fields_list: List[Fields],
    xmin_index_list: np.ndarray,
    xmax_index_list: np.ndarray,
    ymin_index_list: np.ndarray,
    ymax_index_list: np.ndarray,
    npatches: int, nx: int, ny: int, ng: int
):
    """
    Synchronize currents between patches.
    
    Parameters
    ----------
    fields_list : List[Fields]
        List of fields of all patches.
    xmin_index_list : np.ndarray
        Array of indices of patches at the minimum x boundary.
    xmax_index_list : np.ndarray
        Array of indices of patches at the maximum x boundary.
    ymin_index_list : np.ndarray
        Array of indices of patches at the minimum y boundary.
    ymax_index_list : np.ndarray
        Array of indices of patches at the maximum y boundary.
    npatches : int
        Number of patches.
    nx : int
        Number of cells in x direction.
    ny : int
        Number of cells in y direction.
    ng : int
        Number of guard cells.
    """
    pass

def sync_guard_fields_2d(
    fields_list: List[Fields],
    xmin_index_list: np.ndarray,
    xmax_index_list: np.ndarray,
    ymin_index_list: np.ndarray,
    ymax_index_list: np.ndarray,
    npatches: int, nx: int, ny: int, ng: int
):
    """
    Synchronize guard cells between patches for E and B fields.
    
    Parameters
    ----------
    fields_list : List[Fields]
        List of fields of all patches containing E and B fields
    xmin_index_list : np.ndarray
        Array of indices of patches at the minimum x boundary
    xmax_index_list : np.ndarray
        Array of indices of patches at the maximum x boundary
    ymin_index_list : np.ndarray
        Array of indices of patches at the minimum y boundary
    ymax_index_list : np.ndarray
        Array of indices of patches at the maximum y boundary
    npatches : int
        Number of patches
    nx : int
        Number of cells in x direction (excluding guards)
    ny : int
        Number of cells in y direction (excluding guards)
    ng : int
        Number of guard cells
    """
    pass
