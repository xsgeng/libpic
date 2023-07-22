from numba import njit, prange
import numpy as np
from scipy.constants import mu_0, epsilon_0, c, e

@njit
def current_deposit_2d(rho, jx, jy, jz, x, y, uz, inv_gamma, x_old, y_old, pruned, npart, dx, dy, x0, y0, dt, w, q):
    """
    Current deposition in 2D for CPU.

    Parameters
    ----------
    rho : 2D array of floats
        Charge density.
    jx, jy, jz : 2D arrays of floats
        Current density in x, y, z directions.
    x, y : 1D arrays of floats
        Particle positions.
    uz : 1D array of floats
        Particle velocities.
    inv_gamma : 1D array of floats
        Particle inverse gamma.
    x_old, y_old : 1D arrays of floats
        Particle positions at t + dt/2.
    pruned : 1D array of booleans
        Boolean array indicating if the particle has been pruned.
    npart : int
        Number of particles.
    dx, dy : floats
        Cell sizes in x and y directions.
    dt : float
        Time step.
    w : 1D array of floats
        Particle weights.
    q : float
        Charge of the particles.
    """
    for ip in range(npart):
        if pruned[ip]:
            continue
        vz = uz[ip]*c*inv_gamma[ip]
        current(rho, jx, jy, jz, x[ip]-x0, y[ip]-y0, vz, x_old[ip]-x0, y_old[ip]-y0, dx, dy, dt, w[ip], q)

@njit(boundscheck=False)
def current(
    rho, jx, jy, jz, 
    x, y, vz,
    x_old, y_old,
    dx, dy, dt,
    w, q,
):
    """ 
    Compute the current through the charge distribution.

    The following code is not fully optimized, but is written to be readable.

    Parameters
    ----------
    rho : 2D array of floats
        Charge density.
    jx, jy, jz : 2D arrays of floats
        Current density in x, y, z directions.
    x, y : scalar floats
        Particle position of the current particle relative to the current patch.
    vz : scalar float
        Particle velocity.
    x_old, y_old : scalar floats
        Particle positions at t + dt/2.
    dx, dy : floats
        Cell sizes in x and y directions.
    dt : float
        Time step.
    w : scalar float
        Particle weight.
    q : float
        Charge of the particles.

    """

    # positions at t + dt/2, before pusher
    # +0.5 for cell-centered coordinate
    x_over_dx = x_old / dx
    ix0 = int(np.floor(x_over_dx+0.5))
    y_over_dy = y_old / dy
    iy0 = int(np.floor(y_over_dy+0.5))

    S0x = S(x_over_dx, 0)
    S0y = S(y_over_dy, 0)

    # positions at t + 3/2*dt, after pusher
    x_over_dx = x / dx
    ix1 = int(np.floor(x_over_dx+0.5))
    dcell_x = ix1 - ix0

    y_over_dy = y / dy
    iy1 = int(np.floor(y_over_dy+0.5))
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

    # i and j are the relative shift, 0-based index
    # [0,   1, 2, 3, 4]
    #     [-1, 0, 1, 2] for dcell = 1;
    #     [-1, 0, 1] for dcell_ = 0
    # [-2, -1, 0, 1] for dcell = -1
    for j in range(min(1, 1+dcell_y), max(4, 4+dcell_y)):
        jx_ = 0.0
        iy = iy0 + (j - 2)
        for i in range(min(1, 1+dcell_x), max(4, 4+dcell_x)):
            ix = ix0 + (i - 2)
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
