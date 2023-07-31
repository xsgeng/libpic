from numba import njit, prange, get_num_threads, get_thread_id
import numpy as np
from scipy.constants import mu_0, epsilon_0, c, e

nbuff = 64

@njit(cache=True)
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
            vx = ux[ipart_global]*c*inv_gamma[ipart_global]
            vy = uy[ipart_global]*c*inv_gamma[ipart_global]
            vz = uz[ipart_global]*c*inv_gamma[ipart_global]
            x_old = x[ipart_global] - vx*0.5*dt - x0
            y_old = y[ipart_global] - vy*0.5*dt - y0
            x_adv = x[ipart_global] + vx*0.5*dt - x0
            y_adv = y[ipart_global] + vy*0.5*dt - y0

            # positions at t + dt/2, before pusher
            # +0.5 for cell-centered coordinate
            x_over_dx0 = x_old / dx
            ix0 = int(np.floor(x_over_dx0+0.5))
            y_over_dy0 = y_old / dy
            iy0 = int(np.floor(y_over_dy0+0.5))

            delta = x_over_dx0 - ix0
            delta2 = delta * delta
            S0x[0, ip] = 0.0
            S0x[1, ip] = 0.5 * ( delta2+delta+0.25 )
            S0x[2, ip] = 0.75 - delta2
            S0x[3, ip] = 0.5 * ( delta2-delta+0.25 )
            S0x[4, ip] = 0.0

            delta = y_over_dy0 - iy0
            delta2 = delta * delta
            S0y[0, ip] = 0.0
            S0y[1, ip] = 0.5 * ( delta2+delta+0.25 )
            S0y[2, ip] = 0.75 - delta2
            S0y[3, ip] = 0.5 * ( delta2-delta+0.25 )
            S0y[4, ip] = 0.0

            # positions at t + 3/2*dt, after pusher
            x_over_dx1 = x_adv / dx
            ix1 = int(np.floor(x_over_dx1+0.5))
            dcell_x = ix1 - ix0

            y_over_dy1 = y_adv / dy
            iy1 = int(np.floor(y_over_dy1+0.5))
            dcell_y = iy1 - iy0

            delta = x_over_dx0 - ix0
            delta2 = delta * delta
            delta_minus = 0.5 * ( delta2+delta+0.25 )
            delta_mid = 0.75 - delta2
            delta_positive = 0.5 * ( delta2-delta+0.25 )

            minus = dcell_x == -1
            mid = dcell_x == 0
            positive = dcell_x == 1
            S1x[0, ip] = minus * delta_minus
            S1x[1, ip] = minus * delta_mid      + mid * delta_minus
            S1x[2, ip] = minus * delta_positive + mid * delta_mid      + positive * delta_minus
            S1x[3, ip] =                          mid * delta_positive + positive * delta_mid
            S1x[4, ip] =                                                 positive * delta_positive

            delta = y_over_dy0 - iy0
            delta2 = delta * delta
            delta_minus = 0.5 * ( delta2+delta+0.25 )
            delta_mid = 0.75 - delta2
            delta_positive = 0.5 * ( delta2-delta+0.25 )

            minus = dcell_y == -1
            mid = dcell_y == 0
            positive = dcell_y == 1
            S1y[0, ip] = minus * delta_minus
            S1y[1, ip] = minus * delta_mid      + mid * delta_minus
            S1y[2, ip] = minus * delta_positive + mid * delta_mid      + positive * delta_minus
            S1y[3, ip] =                          mid * delta_positive + positive * delta_mid
            S1y[4, ip] =                                                 positive * delta_positive

            for i in range(5):
                DSx[i, ip] = S1x[i, ip] - S0x[i, ip]
                DSy[i, ip] = S1y[i, ip] - S0y[i, ip]
                jy_buff[i, ip] = 0

        for ip in range(npart_buff):
            one_third = 1.0 / 3.0
            charge_density = q * w[ibuff+ip] / (dx*dy) 
            charge_density *= ~pruned[ibuff+ip]
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
                    jz[ix, iy] += factor * wz * vz
                    rho[ix, iy] += charge_density * S1x[i, ip] * S1y[j, ip]


def test_current():
    from time import perf_counter_ns

    nx = 64
    ny = 64
    npart = 1000000
    x0 = 0.0
    y0 = 0.0
    dx = 1.0e-6
    dy = 1.0e-6
    lx = nx * dx
    ly = ny * dy
    dt = dx / c / 2
    q = e
    w = np.ones(npart)
    x = np.random.uniform(low=3*dx, high=lx-3*dx, size=npart)
    y = np.random.uniform(low=3*dy, high=ly-3*dy, size=npart)
    ux = np.random.uniform(low=-1.0, high=1.0, size=npart)
    uy = np.random.uniform(low=-1.0, high=1.0, size=npart)
    uz = np.random.uniform(low=-1.0, high=1.0, size=npart)
    inv_gamma = 1 / np.sqrt(1 + ux**2 + uy**2 + uz**2)

    rho = np.zeros((nx, ny))
    jx = np.zeros((nx, ny))
    jy = np.zeros((nx, ny))
    jz = np.zeros((nx, ny))

    pruned = np.full(npart, False)

    current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)
    tic = perf_counter_ns()
    current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)
    toc = perf_counter_ns()
    print(f"current_deposit_2d {(toc - tic)/1e6} ms")
    print(f"current_deposit_2d {(toc - tic)/npart} ns per particle")

if __name__ ==  "__main__":
    test_current()
