from numba import njit, prange, get_num_threads, get_thread_id
import numpy as np
from scipy.constants import mu_0, epsilon_0, c, e

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
    # nthread = get_num_threads()
    S0x = np.zeros(5)
    S1x = np.zeros(5)
    S0y = np.zeros(5)
    S1y = np.zeros(5)
    jy_buff = np.zeros(5)
    
    for ip in range(npart):
        if pruned[ip]:
            continue
        vx = ux[ip]*c*inv_gamma[ip]
        vy = uy[ip]*c*inv_gamma[ip]
        vz = uz[ip]*c*inv_gamma[ip]
        x_old = x[ip] - vx*0.5*dt
        y_old = y[ip] - vy*0.5*dt
        x_adv = x[ip] + vx*0.5*dt
        y_adv = y[ip] + vy*0.5*dt
        current(rho, jx, jy, jz, x_adv-x0, y_adv-y0, vz, x_old-x0, y_old-y0, dx, dy, dt, w[ip], q,
                S0x, S0y, S1x, S1y, jy_buff)

@njit(inline="always")
def current(
    rho, jx, jy, jz, 
    x, y, vz,
    x_old, y_old,
    dx, dy, dt,
    w, q,
    S0x, S0y, S1x, S1y, jy_buff
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

    S0x[:] = S(x_over_dx - ix0, 0)
    S0y[:] = S(y_over_dy - iy0, 0)

    # positions at t + 3/2*dt, after pusher
    x_over_dx = x / dx
    ix1 = int(np.floor(x_over_dx+0.5))
    dcell_x = ix1 - ix0

    y_over_dy = y / dy
    iy1 = int(np.floor(y_over_dy+0.5))
    dcell_y = iy1 - iy0

    S1x[:] = S(x_over_dx - ix1, dcell_x)
    S1y[:] = S(y_over_dy - iy1, dcell_y)

    DSx = S1x - S0x
    DSy = S1y - S0y

    one_third = 1.0 / 3.0
    charge_density = q * w / (dx*dy)
    factor = charge_density / dt
    jy_buff[:] = 0

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
            wx = DSx[i] * (S0y[j] + 0.5 * DSy[j])
            wy = DSy[j] * (S0x[i] + 0.5 * DSx[i])
            wz = S0x[i] * S0y[j] + 0.5 * DSx[i] * S0y[j] \
                + 0.5 * S0x[i] * DSy[j] + one_third * DSx[i] * DSy[j]
            
            jx_buff -= factor * dx * wx
            jy_buff[i] -= factor * dy * wy

            jx[ix, iy] += jx_buff
            jy[ix, iy] += jy_buff[i]
            jz[ix, iy] += factor * wz * vz
            # rho[ix, iy] += charge_density * S1x[i] * S1y[j]

@njit(inline="always")
def S(delta, shift):
    delta2 = delta**2

    if shift == 0:
        return (
            0.0,
            0.5 * ( delta2+delta+0.25 ),
            0.75 - delta2,
            0.5 * ( delta2-delta+0.25 ),
            0.0,
        )
    if shift == 1:
        return (
            0.0,
            0.0,
            0.5 * ( delta2+delta+0.25 ),
            0.75 - delta2,
            0.5 * ( delta2-delta+0.25 ),
        )
    if shift == -1:
        return (
            0.5 * ( delta2+delta+0.25 ),
            0.75 - delta2,
            0.5 * ( delta2-delta+0.25 ),
            0.0,
            0.0,
        )


def test_current():
    from time import perf_counter_ns

    npart = 100000
    nx = 100
    ny = 100
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

if __name__ ==  "__main__":
    test_current()
