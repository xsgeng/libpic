from numba import njit, prange
import numpy as np
from scipy.constants import mu_0, epsilon_0, c, e

@njit
def current_deposit_2d(rho, jx, jy, jz, x, y, uz, inv_gamma, x_old, y_old, pruned, npart, dx, dy, dt, w, q):
    for ip in range(npart):
        if pruned[ip]:
            continue
        vz = uz[ip]*c*inv_gamma[ip]
        current(rho, jx, jy, jz, x[ip], y[ip], vz, x_old[ip], y_old[ip], dx, dy, dt, w[ip], q)

@njit(boundscheck=False)
def current(
    rho, jx, jy, jz, 
    x, y, vz,
    x_old, y_old,
    dx, dy, dt,
    w, q,
):
    # positions at t + dt/2, before pusher
    x_over_dx = x_old / dx
    ix0 = int(np.floor(x_over_dx))
    y_over_dy = y_old / dy
    iy0 = int(np.floor(y_over_dy))

    S0x = S(x_over_dx, 0)
    S0y = S(y_over_dy, 0)

    # positions at t + 3/2*dt, after pusher
    x_over_dx = x / dx
    ix1 = int(np.floor(x_over_dx))
    dcell_x = ix1 - ix0

    y_over_dy = y / dy
    iy1 = int(np.floor(y_over_dy))
    dcell_y = iy1 - iy0

    S1x = S(x_over_dx, dcell_x)
    S1y = S(y_over_dy, dcell_y)

    DSx = S1x - S0x
    DSy = S1y - S0y

    one_third = 1.0 / 3.0
    charge_density = q * w / (dx*dy)
    factor = charge_density / dt
    jx_ = 0.0
    jy_ = np.zeros(5)
    jz_ = 0.0
    for j in prange(min(1, 1-dcell_y), max(4, 4+dcell_y)):
        jx_ = 0.0
        iy = iy0 + j - 2
        for i in range(min(1, 1-dcell_x), max(4, 4+dcell_x)):
            ix = ix0 + i - 2
            wx = DSx[i] * (S0y[j] + 0.5 * DSy[j])
            wy = DSy[j] * (S0x[i] + 0.5 * DSx[i])
            wz = S0x[i] * S0y[j] + 0.5 * DSx[i] * S0y[j] \
                + 0.5 * S0x[i] * DSy[j] + one_third * DSx[i] * DSy[j]
            
            jx_ -= factor * dx * wx
            jy_[i] -= factor * dy * wy
            jz_ = factor * wz * vz

            jx[ix, iy] += jx_
            jy[ix, iy] += jy_[i]
            jz[ix, iy] += jz_
            rho[ix, iy] += charge_density * S1x[i] * S1y[j]


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
