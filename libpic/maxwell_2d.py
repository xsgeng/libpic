from numba import njit, prange
import numpy as np
from scipy.constants import mu_0, epsilon_0, c

@njit
def update_efield_2d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    dx, dy, dt, 
    nx, ny, n_guard
):
    bfactorx = dt * c**2 / dx
    bfactory = dt * c**2 / dy
    jfactor = dt / epsilon_0
    for j in range(-1, ny):
        for i in range(-1, nx):
            ex[i, j] += bfactory *  (bz[i, j] - bz[i, j-1]) - jfactor * jx[i, j]
            ey[i, j] += bfactorx * -(bz[i, j] - bz[i-1, j]) - jfactor * jy[i, j]
            ez[i, j] += bfactorx *  (by[i, j] - by[i-1, j]) - bfactory * (bx[i, j] - bx[i, j-1]) - jfactor * jz[i, j]


@njit
def update_bfield_2d(
    ex, ey, ez, 
    bx, by, bz, 
    dx, dy, dt, 
    nx, ny, n_guard
):
    dt_over_dx = dt/dx
    dt_over_dy = dt/dy
    for j in range(-1, ny):
        for i in range(-1, nx):
            bx[i, j] -= dt_over_dy *  (ez[i, j+1] - ez[i, j])
            by[i, j] -= dt_over_dx * -(ez[i+1, j] - ez[i, j])
            bz[i, j] -= dt_over_dx *  (ey[i+1, j] - ey[i, j]) - dt_over_dy * (ex[i, j+1] - ex[i, j])
