from numba import njit, prange
import numpy as np
from scipy.constants import mu_0, epsilon_0, c

@njit(cache=True)
def update_efield_2d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    dx, dy, dt, 
    nx, ny, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for j in range(ny):
        for i in range(nx):
            ex[i, j] += bfactor * ( (bz[i, j] - bz[i, j-1]) / dy) - jfactor * jx[i, j]
            ey[i, j] += bfactor * (-(bz[i, j] - bz[i-1, j]) / dx) - jfactor * jy[i, j]
            ez[i, j] += bfactor * ( (by[i, j] - by[i-1, j]) / dx - (bx[i, j] - bx[i, j-1]) / dy) - jfactor * jz[i, j]


@njit(cache=True)
def update_bfield_2d(
    ex, ey, ez, 
    bx, by, bz, 
    dx, dy, dt, 
    nx, ny, n_guard
):
    for j in range(ny):
        for i in range(nx):
            bx[i, j] -= dt * ( (ez[i, j+1] - ez[i, j]) / dy)
            by[i, j] -= dt * (-(ez[i+1, j] - ez[i, j]) / dx)
            bz[i, j] -= dt * ( (ey[i+1, j] - ey[i, j]) / dx - (ex[i, j+1] - ex[i, j]) / dy)

@njit(cache=True, parallel=True)
def update_efield_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]

        update_efield_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, dt, nx, ny, n_guard)


@njit(cache=True, parallel=True)
def update_bfield_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        update_bfield_2d(ex, ey, ez, bx, by, bz, dx, dy, dt, nx, ny, n_guard)