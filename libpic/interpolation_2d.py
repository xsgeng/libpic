import numpy as np
from numba import njit, prange

from scipy.constants import mu_0, epsilon_0, c, e

@njit(inline="always")
def interpolation_2d(
    x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
    ex, ey, ez, bx, by, bz,
    dx, dy, x0, y0,
    pruned,
):
    gx = np.zeros(3)
    gy = np.zeros(3)
    hx = np.zeros(3)
    hy = np.zeros(3)
    for ip in range(npart):
        if pruned[ip]:
            continue
        ex_part[ip], ey_part[ip], ez_part[ip], bx_part[ip], by_part[ip], bz_part[ip] = \
            em_to_particle(ex, ey, ez, bx, by, bz, x[ip]-x0, y[ip]-y0, dx, dy,
                           gx, gy, hx, hy)

@njit(inline="always")
def em_to_particle(ex, ey, ez, bx, by, bz, x, y, dx, dy, gx, gy, hx, hy):
    x_over_dx = x / dx
    y_over_dy = y / dy

    ix1 = int(np.floor(x_over_dx+0.5))
    gx[:] = get_gx(ix1 - x_over_dx)

    ix2 = int(np.floor(x_over_dx))
    hx[:] = get_gx(ix2 - x_over_dx + 0.5)


    iy1 = int(np.floor(y_over_dy+0.5))
    gy[:] = get_gx(iy1 - y_over_dy)

    iy2 = int(np.floor(y_over_dy))
    hy[:] = get_gx(iy2 - y_over_dy + 0.5)

    ex_part = interp_ex(ex, hx, gy, ix2, iy1)
    ey_part = interp_ey(ey, gx, hy, ix1, iy2)
    ez_part = interp_ez(ez, gx, gy, ix1, iy1)

    bx_part = interp_bx(bx, gx, hy, ix1, iy2)
    by_part = interp_by(by, hx, gy, ix2, iy1)
    bz_part = interp_bz(bz, hx, hy, ix2, iy2)

    return ex_part, ey_part, ez_part, bx_part, by_part, bz_part

@njit
def get_gx(delta):
    delta2 = delta*delta
    return (
        0.5*(0.25 + delta2 + delta),
        0.75 - delta2,
        0.5*(0.25 + delta2 - delta),
    )

@njit
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

@njit
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

@njit
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


@njit
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

@njit
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

@njit
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