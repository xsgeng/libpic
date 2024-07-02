import numpy as np
from numba import njit

subsize = 64
@njit(cache=True)
def interpolation_2d(
    x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
    ex, ey, ez, bx, by, bz,
    dx, dy, x0, y0,
    is_dead,
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
