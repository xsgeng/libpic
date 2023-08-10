import numpy as np
from numba import njit, prange

from scipy.constants import mu_0, epsilon_0, c, e

subsize = 32
@njit(cache=True)
def interpolation_2d(
    x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
    ex, ey, ez, bx, by, bz,
    dx, dy, x0, y0,
    pruned,
):
    gx = np.zeros((3, subsize))
    gy = np.zeros((3, subsize))
    hx = np.zeros((3, subsize))
    hy = np.zeros((3, subsize))

    for ivect in range(0, npart, subsize):
        npart_vec = min(subsize, npart - ivect)
        for ip in range(npart_vec):
            ipart_global = ivect + ip
            x_over_dx = (x[ipart_global] - x0) / dx
            y_over_dy = (y[ipart_global] - y0) / dy

            ix1 = int(np.floor(x_over_dx+0.5))
            get_gx(ix1 - x_over_dx, gx[:, ip])

            ix2 = int(np.floor(x_over_dx))
            get_gx(ix2 - x_over_dx + 0.5, hx[:, ip])

            iy1 = int(np.floor(y_over_dy+0.5))
            get_gx(iy1 - y_over_dy, gy[:, ip])

            iy2 = int(np.floor(y_over_dy))
            get_gx(iy2 - y_over_dy + 0.5, hy[:, ip])

        # for ip in range(npart_vec):
        #     ipart_global = ivect + ip
            if not pruned[ipart_global]:
                ex_part[ipart_global] = interp_ex(ex, hx[:, ip], gy[:, ip], ix2, iy1)
                ey_part[ipart_global] = interp_ey(ey, gx[:, ip], hy[:, ip], ix1, iy2)
                ez_part[ipart_global] = interp_ez(ez, gx[:, ip], gy[:, ip], ix1, iy1)

                bx_part[ipart_global] = interp_bx(bx, gx[:, ip], hy[:, ip], ix1, iy2)
                by_part[ipart_global] = interp_by(by, hx[:, ip], gy[:, ip], ix2, iy1)
                bz_part[ipart_global] = interp_bz(bz, hx[:, ip], hy[:, ip], ix2, iy2)


@njit(cache=True, inline="always")
def get_gx(delta, gx):
    delta2 = delta*delta
    gx[0] = 0.5*(0.25 + delta2 + delta)
    gx[1] = 0.75 - delta2
    gx[2] = 0.5*(0.25 + delta2 - delta)

@njit(cache=True)
def interp_ex(ex, hx, gy, ix2, iy1):
    ex_part = \
          gy[ 0] * (hx[ 0] * ex[ix2-1,iy1-1] \
        +           hx[ 1] * ex[ix2  ,iy1-1] \
        +           hx[ 2] * ex[ix2+1,iy1-1]) \
        + gy[ 1] * (hx[ 0] * ex[ix2-1,iy1  ] \
        +           hx[ 1] * ex[ix2  ,iy1  ] \
        +           hx[ 2] * ex[ix2+1,iy1  ]) \
        + gy[ 2] * (hx[ 0] * ex[ix2-1,iy1+1] \
        +           hx[ 1] * ex[ix2  ,iy1+1] \
        +           hx[ 2] * ex[ix2+1,iy1+1])
    return ex_part

@njit(cache=True)
def interp_ey(ey, gx, hy, ix1, iy2):
    ey_part = \
          hy[ 0] * (gx[ 0] * ey[ix1-1,iy2-1] \
        +           gx[ 1] * ey[ix1  ,iy2-1] \
        +           gx[ 2] * ey[ix1+1,iy2-1]) \
        + hy[ 1] * (gx[ 0] * ey[ix1-1,iy2  ] \
        +           gx[ 1] * ey[ix1  ,iy2  ] \
        +           gx[ 2] * ey[ix1+1,iy2  ]) \
        + hy[ 2] * (gx[ 0] * ey[ix1-1,iy2+1] \
        +           gx[ 1] * ey[ix1  ,iy2+1] \
        +           gx[ 2] * ey[ix1+1,iy2+1])
    return ey_part

@njit(cache=True)
def interp_ez(ez, gx, gy, ix1, iy1):
    ez_part = \
          gy[ 0] * (gx[ 0] * ez[ix1-1,iy1-1] \
        +           gx[ 1] * ez[ix1  ,iy1-1] \
        +           gx[ 2] * ez[ix1+1,iy1-1]) \
        + gy[ 1] * (gx[ 0] * ez[ix1-1,iy1  ] \
        +           gx[ 1] * ez[ix1  ,iy1  ] \
        +           gx[ 2] * ez[ix1+1,iy1  ]) \
        + gy[ 2] * (gx[ 0] * ez[ix1-1,iy1+1] \
        +           gx[ 1] * ez[ix1  ,iy1+1] \
        +           gx[ 2] * ez[ix1+1,iy1+1])
    return ez_part


@njit(cache=True)
def interp_bx(bx, gx, hy, ix1, iy2):
    bx_part = \
          hy[ 0] * (gx[ 0] * bx[ix1-1,iy2-1] \
        +           gx[ 1] * bx[ix1  ,iy2-1] \
        +           gx[ 2] * bx[ix1+1,iy2-1]) \
        + hy[ 1] * (gx[ 0] * bx[ix1-1,iy2  ] \
        +           gx[ 1] * bx[ix1  ,iy2  ] \
        +           gx[ 2] * bx[ix1+1,iy2  ]) \
        + hy[ 2] * (gx[ 0] * bx[ix1-1,iy2+1] \
        +           gx[ 1] * bx[ix1  ,iy2+1] \
        +           gx[ 2] * bx[ix1+1,iy2+1])
    return bx_part

@njit(cache=True)
def interp_by(by, hx, gy, ix2, iy1):
    bx_part = \
          gy[ 0] * (hx[ 0] * by[ix2-1,iy1-1] \
        +           hx[ 1] * by[ix2  ,iy1-1] \
        +           hx[ 2] * by[ix2+1,iy1-1]) \
        + gy[ 1] * (hx[ 0] * by[ix2-1,iy1  ] \
        +           hx[ 1] * by[ix2  ,iy1  ] \
        +           hx[ 2] * by[ix2+1,iy1  ]) \
        + gy[ 2] * (hx[ 0] * by[ix2-1,iy1+1] \
        +           hx[ 1] * by[ix2  ,iy1+1] \
        +           hx[ 2] * by[ix2+1,iy1+1])
    return bx_part

@njit(cache=True)
def interp_bz(bz, hx, hy, ix2, iy2):
    bx_part = \
          hy[ 0] * (hx[ 0] * bz[ix2-1,iy2-1] \
        +           hx[ 1] * bz[ix2  ,iy2-1] \
        +           hx[ 2] * bz[ix2+1,iy2-1]) \
        + hy[ 1] * (hx[ 0] * bz[ix2-1,iy2  ] \
        +           hx[ 1] * bz[ix2  ,iy2  ] \
        +           hx[ 2] * bz[ix2+1,iy2  ]) \
        + hy[ 2] * (hx[ 0] * bz[ix2-1,iy2+1] \
        +           hx[ 1] * bz[ix2  ,iy2+1] \
        +           hx[ 2] * bz[ix2+1,iy2+1])
    return bx_part


def test_interp():
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
    ux= np.random.uniform(low=-1.0, high=1.0, size=npart)
    uy= np.random.uniform(low=-1.0, high=1.0, size=npart)
    uz= np.random.uniform(low=-1.0, high=1.0, size=npart)
    inv_gamma = 1 / np.sqrt(1 + ux**2 + uy**2 + uz**2)

    ex_part = np.random.uniform(low=-1.0, high=1.0, size=npart)
    ey_part = np.random.uniform(low=-1.0, high=1.0, size=npart)
    ez_part = np.random.uniform(low=-1.0, high=1.0, size=npart)
    bx_part = np.random.uniform(low=-1.0, high=1.0, size=npart)
    by_part = np.random.uniform(low=-1.0, high=1.0, size=npart)
    bz_part = np.random.uniform(low=-1.0, high=1.0, size=npart)

    ex = np.zeros((nx, ny))
    ey = np.zeros((nx, ny))
    ez = np.zeros((nx, ny))
    bx = np.zeros((nx, ny))
    by = np.zeros((nx, ny))
    bz = np.zeros((nx, ny))

    pruned = np.full(npart, False)

    interpolation_2d(
        x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
        ex, ey, ez, bx, by, bz,
        dx, dy, x0, y0,
        pruned,
    )
    tic = perf_counter_ns()
    interpolation_2d(
        x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
        ex, ey, ez, bx, by, bz,
        dx, dy, x0, y0,
        pruned,
    )
    toc = perf_counter_ns()
    print(f"current_deposit_2d {(toc - tic)/1e6} ms")

if __name__ ==  "__main__":
    test_interp()