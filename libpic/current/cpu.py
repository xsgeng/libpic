import numpy as np
from numba import njit
from scipy.constants import c

@njit(inline="always")
def calculate_S(delta, shift, ip, S):
    delta2 = delta * delta

    delta_minus    = 0.5 * ( delta2+delta+0.25 )
    delta_mid      = 0.75 - delta2
    delta_positive = 0.5 * ( delta2-delta+0.25 )

    minus = shift == -1
    mid = shift == 0
    positive = shift == 1

    S[0, ip] = minus * delta_minus
    S[1, ip] = minus * delta_mid      + mid * delta_minus
    S[2, ip] = minus * delta_positive + mid * delta_mid      + positive * delta_minus
    S[3, ip] =                          mid * delta_positive + positive * delta_mid
    S[4, ip] =                                                 positive * delta_positive


nbuff = 64
@njit
def current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q):
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

    x_old = np.zeros(nbuff)
    y_old = np.zeros(nbuff)
    x_adv = np.zeros(nbuff)
    y_adv = np.zeros(nbuff)
    vz = np.zeros(nbuff)

    S0x = np.zeros((5, nbuff))
    S1x = np.zeros((5, nbuff))
    S0y = np.zeros((5, nbuff))
    S1y = np.zeros((5, nbuff))
    DSx = np.zeros((5, nbuff))
    DSy = np.zeros((5, nbuff))
    jy_buff = np.zeros((5, nbuff))

    for ibuff in range(0, npart, nbuff):
        npart_buff = min(nbuff, npart - ibuff)
        for ip in range(npart_buff):
            ipart_global = ibuff + ip
            if pruned[ipart_global]:
                vz[ip] = 0.0
                x_old[ip] = 0.0
                y_old[ip] = 0.0
                x_adv[ip] = 0.0
                y_adv[ip] = 0.0
                continue
            vx = ux[ipart_global]*c*inv_gamma[ipart_global]
            vy = uy[ipart_global]*c*inv_gamma[ipart_global]
            vz[ip] = uz[ipart_global]*c*inv_gamma[ipart_global] if ~pruned[ipart_global] else 0.0
            x_old[ip] = x[ipart_global] - vx*0.5*dt - x0
            y_old[ip] = y[ipart_global] - vy*0.5*dt - y0
            x_adv[ip] = x[ipart_global] + vx*0.5*dt - x0
            y_adv[ip] = y[ipart_global] + vy*0.5*dt - y0

        for ip in range(npart_buff):
            ipart_global = ibuff + ip
            # positions at t + dt/2, before pusher
            # +0.5 for cell-centered coordinate
            x_over_dx0 = x_old[ip] / dx
            ix0 = int(np.floor(x_over_dx0+0.5))
            y_over_dy0 = y_old[ip] / dy
            iy0 = int(np.floor(y_over_dy0+0.5))

            calculate_S(ix0 - x_over_dx0, 0, ip, S0x) #gx
            calculate_S(iy0 - y_over_dy0, 0, ip, S0y)

            # positions at t + 3/2*dt, after pusher
            x_over_dx1 = x_adv[ip] / dx
            ix1 = int(np.floor(x_over_dx1+0.5))
            dcell_x = ix1 - ix0

            y_over_dy1 = y_adv[ip] / dy
            iy1 = int(np.floor(y_over_dy1+0.5))
            dcell_y = iy1 - iy0

            calculate_S(ix1 - x_over_dx1 , dcell_x, ip, S1x)
            calculate_S(iy1 - y_over_dy1 , dcell_y, ip, S1y)

            for i in range(5):
                DSx[i, ip] = S1x[i, ip] - S0x[i, ip]
                DSy[i, ip] = S1y[i, ip] - S0y[i, ip]
                jy_buff[i, ip] = 0

            one_third = 1.0 / 3.0
            charge_density = q * w[ipart_global] / (dx*dy)
            charge_density *= ~pruned[ipart_global]
            factor = charge_density / dt


            # i and j are the relative shift, 0-based index
            # [0,   1, 2, 3, 4]
            #     [-1, 0, 1, 2] for dcell = 1;
            #     [-1, 0, 1] for dcell_ = 0
            # [-2, -1, 0, 1] for dcell = -1
            for j in range(min(1, 1+dcell_y), max(4, 4+dcell_y)):
                jx_buff = 0.0
                iy = iy0 + (j - 2)
                for i in range(min(1, 1+dcell_x), max(4, 4+dcell_x)):
                    ix = ix0 + (i - 2)

                    wx = DSx[i, ip] * (S0y[j, ip] + 0.5 * DSy[j, ip])
                    wy = DSy[j, ip] * (S0x[i, ip] + 0.5 * DSx[i, ip])
                    wz = S0x[i, ip] * S0y[j, ip] + 0.5 * DSx[i, ip] * S0y[j, ip] \
                        + 0.5 * S0x[i, ip] * DSy[j, ip] + one_third * DSx[i, ip] * DSy[j, ip]

                    jx_buff -= factor * dx * wx
                    jy_buff[i, ip] -= factor * dy * wy

                    jx[ix, iy] += jx_buff
                    jy[ix, iy] += jy_buff[i, ip]
                    jz[ix, iy] += factor*dt * wz * vz[ip]
                    rho[ix, iy] += charge_density * S1x[i, ip] * S1y[j, ip]