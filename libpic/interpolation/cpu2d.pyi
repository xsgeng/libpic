from numpy.typing import NDArray
from numpy import float64, bool_

def interpolation_patches_2d(
    x_list: list[NDArray[float64]], y_list: list[NDArray[float64]],
    ex_part_list: list[NDArray[float64]], ey_part_list: list[NDArray[float64]], ez_part_list: list[NDArray[float64]],
    bx_part_list: list[NDArray[float64]], by_part_list: list[NDArray[float64]], bz_part_list: list[NDArray[float64]],
    is_dead_list: list[NDArray[bool_]],
    ex_list: list[NDArray[float64]], ey_list: list[NDArray[float64]], ez_list: list[NDArray[float64]],
    bx_list: list[NDArray[float64]], by_list: list[NDArray[float64]], bz_list: list[NDArray[float64]],
    x0_list: list[float], y0_list: list[float],
    npatches: int,
    dx: float, dy: float, nx: int, ny: int
) -> None: ...

def _interpolation_2d(
    x: NDArray[float64], y: NDArray[float64], 
    ex_part: NDArray[float64], ey_part: NDArray[float64], ez_part: NDArray[float64], 
    bx_part: NDArray[float64], by_part: NDArray[float64], bz_part: NDArray[float64], 
    is_dead: NDArray[bool_],
    npart: int,
    ex: NDArray[float64], ey: NDArray[float64], ez: NDArray[float64], bx: NDArray[float64], by: NDArray[float64], bz: NDArray[float64],
    dx: float, dy: float, x0: float, y0: float, 
    nx: int, ny: int,
) -> None: ...