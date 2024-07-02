
from numba import njit, prange

from .interpolation_2d import interpolation_2d


@njit(cache=True, parallel=True)
def interpolation_patches_2d(
    x_list, y_list,
    ex_part_list, ey_part_list, ez_part_list,
    bx_part_list, by_part_list, bz_part_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    x0_list, y0_list,
    npatches,
    dx, dy,
    is_dead_list,
) -> None:
    for ipatch in prange(npatches):
        x = x_list[ipatch]
        y = y_list[ipatch]
        ex_part = ex_part_list[ipatch]
        ey_part = ey_part_list[ipatch]
        ez_part = ez_part_list[ipatch]
        bx_part = bx_part_list[ipatch]
        by_part = by_part_list[ipatch]
        bz_part = bz_part_list[ipatch]
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        x0 = x0_list[ipatch]
        y0 = y0_list[ipatch]
        is_dead = is_dead_list[ipatch]
        npart = len(is_dead)
        interpolation_2d(
            x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
            ex, ey, ez, bx, by, bz,
            dx, dy, x0, y0,
            is_dead,
        )
