
from numba import njit, prange

from .interpolation_2d import interpolation_2d


@njit(cache=True, parallel=True)
def interpolation(
    x_list, y_list,
    ex_part_list, ey_part_list, ez_part_list,
    bx_part_list, by_part_list, bz_part_list,
    npart_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    xaxis_list, yaxis_list,
    npatches,
    dx, dy,
    pruned_list,
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
        x0 = xaxis_list[ipatch][0]
        y0 = yaxis_list[ipatch][0]
        x = x_list[ipatch]
        y = y_list[ipatch]
        pruned = pruned_list[ipatch]
        npart = len(pruned)
        interpolation_2d(
            x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
            ex, ey, ez, bx, by, bz,
            dx, dy, x0, y0,
            pruned,
        )