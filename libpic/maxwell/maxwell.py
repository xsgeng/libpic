from numba import njit, prange
import numpy as np
from scipy.constants import mu_0, epsilon_0, c

@njit(parallel=True)
def update_efield_3d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    dx, dy, dz, dt, 
    nx, ny, nz, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                ex[i, j, k] += bfactor * ((bz[i, j, k] - bz[i, j-1, k]) / dy - (by[i, j, k] - by[i, j, k-1]) / dz) - jfactor * jx[i, j ,k]
                ey[i, j, k] += bfactor * ((bx[i, j, k] - bx[i, j, k-1]) / dz - (bz[i, j, k] - bz[i-1, j, k]) / dx) - jfactor * jy[i, j, k]
                ez[i, j, k] += bfactor * ((by[i, j, k] - by[i-1, j, k]) / dx - (bx[i, j, k] - bx[i, j-1, k]) / dy) - jfactor * jz[i, j, k]


@njit(parallel=True)
def update_bfield_3d(
    ex, ey, ez, bx, by, bz, 
    dx, dy, dz, dt, nx, ny, nz, n_guard
):
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                bx[i, j, k] -= dt * ((ez[i, j+1, k] - ez[i, j, k]) / dy - (ey[i, j, k+1] - ey[i, j, k]) / dz)
                by[i, j, k] -= dt * ((ex[i, j, k+1] - ex[i, j, k]) / dz - (ez[i+1, j, k] - ez[i, j, k]) / dx)
                bz[i, j, k] -= dt * ((ey[i+1, j, k] - ey[i, j, k]) / dx - (ex[i, j+1, k] - ex[i, j, k]) / dy)