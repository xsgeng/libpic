from numba import njit, prange
from scipy.constants import c

from .boris import boris


@njit(cache=True, parallel=True)
def boris_push_patches(
    ux_list, uy_list, uz_list, inv_gamma_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    pruned_list,
    npatches, q, m, dt
) -> None:
    for ipatch in prange(npatches):
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]

        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        pruned = pruned_list[ipatch]
        npart = len(pruned)
        boris( ux, uy, uz, inv_gamma, ex, ey, ez, bx, by, bz, q, m, npart, pruned, dt )


@njit(cache=True)
def push_position_2d( x, y, ux, uy, inv_gamma, N, pruned, dt):
    """
    Advance the particles' positions over `dt` using the momenta `ux`, `uy`, `uz`,
    """
    # Timestep, multiplied by c
    cdt = c*dt

    for ip in prange(N) :
        if pruned[ip]:
            continue
        x[ip] += cdt * inv_gamma[ip] * ux[ip]
        y[ip] += cdt * inv_gamma[ip] * uy[ip]

@njit(cache=True, parallel=True)
def push_position_patches_2d(
    x_list, y_list,
    ux_list, uy_list, inv_gamma_list,
    pruned_list, 
    npatches, dt,
) -> None:
    for ipatch in prange(npatches):
        x = x_list[ipatch]
        y = y_list[ipatch]

        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]

        pruned = pruned_list[ipatch]
        npart = len(pruned)
        push_position_2d( x, y, ux, uy, inv_gamma, npart, pruned, dt )