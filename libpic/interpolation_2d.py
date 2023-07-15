import numpy as np
from numba import njit, prange


@njit(parallel=True)
def interpolation_2d(
    x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, N,
    ex, ey, ez, bx, by, bz,
    dx, dy,
    pruned,
):
    for ip in prange(N):
        if pruned[ip]:
            continue
        ex_part[ip], ey_part[ip], ez_part[ip], bx_part[ip], by_part[ip], bz_part[ip] = \
            em_to_particle(ex, ey, ez, bx, by, bz, x[ip], y[ip], dx, dy)

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

@njit
def em_to_particle(ex, ey, ez, bx, by, bz, x, y, dx, dy):
    x_over_dx = x / dx
    y_over_dy = y / dy

    ix1 = int(np.floor(x_over_dx+0.5))
    gx = get_gx(ix1 - x_over_dx)

    ix2 = int(np.floor(x_over_dx))
    hx = get_gx(ix2 - x_over_dx + 0.5)


    iy1 = int(np.floor(y_over_dy+0.5))
    gy = get_gx(iy1 - y_over_dy)

    iy2 = int(np.floor(y_over_dy))
    hy = get_gx(iy2 - y_over_dy + 0.5)

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
    return [
        0.5*(0.25 + delta2 + delta),
        0.75 - delta2,
        0.5*(0.25 + delta2 - delta),
    ]
