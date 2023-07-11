from numba import njit, prange
import numpy as np
from scipy.constants import mu_0, epsilon_0, c, e

@njit
def current_deposit_3d(rho, jx, jy, jz, x, y, z, x_old, y_old, z_old, pruned, npart, dx, dy, dz, dt, w, q):
    for ip in range(npart):
        if pruned[ip]:
            continue
        current_3d(rho, jx, jy, jz, x[ip], y[ip], z[ip], x_old[ip], y_old[ip], z_old[ip], dx, dy, dz, dt, w[ip], q)

@njit(boundscheck=False)
def current_3d(
    rho, jx, jy, jz, 
    x, y, z, 
    x_old, y_old, z_old,
    dx, dy, dz, dt,
    w, q,
):
    # positions at t + dt/2, before pusher
    x_over_dx = x_old / dx
    ix0 = int(np.floor(x_over_dx))
    y_over_dy = y_old / dy
    iy0 = int(np.floor(y_over_dy))
    z_over_dz = z_old / dz
    iz0 = int(np.floor(z_over_dz))

    S0x = S(x_over_dx, 0)
    S0y = S(y_over_dy, 0)
    S0z = S(z_over_dz, 0)

    # positions at t + 3/2*dt, after pusher
    x_over_dx = x / dx
    ix1 = int(np.floor(x_over_dx))
    dcell_x = ix1 - ix0

    y_over_dy = y / dy
    iy1 = int(np.floor(y_over_dy))
    dcell_y = iy1 - iy0

    z_over_dz = z / dz
    iz1 = int(np.floor(z_over_dz))
    dcell_z = iz1 - iz0

    S1x = S(x_over_dx, dcell_x)
    S1y = S(y_over_dy, dcell_y)
    S1z = S(z_over_dz, dcell_z)

    DSx = S1x - S0x
    DSy = S1y - S0y
    DSz = S1z - S0z

    one_third = 1.0 / 3.0
    charge_density = q * w / (dx*dy*dz)
    factor = charge_density / dt
    jx_ = 0.0
    jy_ = np.zeros(5)
    jz_ = np.zeros((5, 5))
    for k in range(min(1, 1-dcell_z), max(4, 4+dcell_z)):
        jy_[:] = 0.0
        iz = iz0 + k - 2
        for j in range(min(1, 1-dcell_y), max(4, 4+dcell_y)):
            jx_ = 0.0
            iy = iy0 + j - 2
            for i in range(min(1, 1-dcell_x), max(4, 4+dcell_x)):
                ix = ix0 + i - 2
                wx = DSx[i] * (S0y[j] * S0z[k] + 0.5 * DSy[j] * S0z[k] +
                               0.5 * S0y[j] * DSz[k] + one_third * DSy[j] * DSz[k])
                wy = DSy[j] * (S0x[i] * S0z[k] + 0.5 * DSx[i] * S0z[k] +
                               0.5 * S0x[i] * DSz[k] + one_third * DSx[i] * DSz[k])
                wz = DSz[k] * (S0x[i] * S0y[j] + 0.5 * DSx[i] * S0y[j] +
                               0.5 * S0x[i] * DSy[j] + one_third * DSx[i] * DSy[j])
                
                jx_ -= factor * dx * wx
                jy_[i] -= factor * dy * wy
                jz_[i, j] -= factor * dz * wz

                jx[ix, iy, iz] += jx_
                jy[ix, iy, iz] += jy_[i]
                jz[ix, iy, iz] += jz_[i, j]
                rho[ix, iy, iz] += charge_density * S1x[i] * S1y[j] * S1z[k]


    # for k in range()
    # print(S0x.shape)
    # print(S0z, S1z)


@njit
def S(x_over_dx, shift):
    ix = np.floor(x_over_dx)
    # Xi - x
    delta = ix - x_over_dx
    delta2 = delta**2

    S = np.zeros(5)
    S[shift+1] = 0.5 * ( delta2+delta+0.25 )
    S[shift+2] = 0.75 - delta2
    S[shift+3] = 0.5 * ( delta2-delta+0.25 )
    return S
