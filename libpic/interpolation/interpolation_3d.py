import numpy as np
from numba import njit, prange


@njit(parallel=True)
def interpolation_3d(
    x, y, z, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, N,
    ex, ey, ez, bx, by, bz,
    dx, dy, dz,
):
    for ip in prange(N):
        ex_part[ip], ey_part[ip], ez_part[ip], bx_part[ip], by_part[ip], bz_part[ip] = \
            em_to_particle_3d(ex, ey, ez, bx, by, bz, x[ip], y[ip], z[ip], dx, dy, dz)

@njit
def interp_ex_3d(ex, hx, gy, gz, ix2, iy1, iz1):
    ex_part = \
          gz[ 0] * (gy[ 0] * (hx[ 0] * ex[ix2-1,iy1-1,iz1-1] \
        +                     hx[ 1] * ex[ix2  ,iy1-1,iz1-1] \
        +                     hx[ 2] * ex[ix2+1,iy1-1,iz1-1]) \
        +           gy[ 1] * (hx[ 0] * ex[ix2-1,iy1  ,iz1-1] \
        +                     hx[ 1] * ex[ix2  ,iy1  ,iz1-1] \
        +                     hx[ 2] * ex[ix2+1,iy1  ,iz1-1]) \
        +           gy[ 2] * (hx[ 0] * ex[ix2-1,iy1+1,iz1-1] \
        +                     hx[ 1] * ex[ix2  ,iy1+1,iz1-1] \
        +                     hx[ 2] * ex[ix2+1,iy1+1,iz1-1])) \
        + gz[ 1] * (gy[ 0] * (hx[ 0] * ex[ix2-1,iy1-1,iz1  ] \
        +                     hx[ 1] * ex[ix2  ,iy1-1,iz1  ] \
        +                     hx[ 2] * ex[ix2+1,iy1-1,iz1  ]) \
        +           gy[ 1] * (hx[ 0] * ex[ix2-1,iy1  ,iz1  ] \
        +                     hx[ 1] * ex[ix2  ,iy1  ,iz1  ] \
        +                     hx[ 2] * ex[ix2+1,iy1  ,iz1  ]) \
        +           gy[ 2] * (hx[ 0] * ex[ix2-1,iy1+1,iz1  ] \
        +                     hx[ 1] * ex[ix2  ,iy1+1,iz1  ] \
        +                     hx[ 2] * ex[ix2+1,iy1+1,iz1  ])) \
        + gz[ 2] * (gy[ 0] * (hx[ 0] * ex[ix2-1,iy1-1,iz1+1] \
        +                     hx[ 1] * ex[ix2  ,iy1-1,iz1+1] \
        +                     hx[ 2] * ex[ix2+1,iy1-1,iz1+1]) \
        +           gy[ 1] * (hx[ 0] * ex[ix2-1,iy1  ,iz1+1] \
        +                     hx[ 1] * ex[ix2  ,iy1  ,iz1+1] \
        +                     hx[ 2] * ex[ix2+1,iy1  ,iz1+1]) \
        +           gy[ 2] * (hx[ 0] * ex[ix2-1,iy1+1,iz1+1] \
        +                     hx[ 1] * ex[ix2  ,iy1+1,iz1+1] \
        +                     hx[ 2] * ex[ix2+1,iy1+1,iz1+1]))
    return ex_part

@njit
def interp_ey_3d(ey, gx, hy, gz, ix1, iy2, iz1):
    ey_part = \
          gz[ 0] * (hy[ 0] * (gx[ 0] * ey[ix1-1,iy2-1,iz1-1] \
        +                     gx[ 1] * ey[ix1  ,iy2-1,iz1-1] \
        +                     gx[ 2] * ey[ix1+1,iy2-1,iz1-1]) \
        +           hy[ 1] * (gx[ 0] * ey[ix1-1,iy2  ,iz1-1] \
        +                     gx[ 1] * ey[ix1  ,iy2  ,iz1-1] \
        +                     gx[ 2] * ey[ix1+1,iy2  ,iz1-1]) \
        +           hy[ 2] * (gx[ 0] * ey[ix1-1,iy2+1,iz1-1] \
        +                     gx[ 1] * ey[ix1  ,iy2+1,iz1-1] \
        +                     gx[ 2] * ey[ix1+1,iy2+1,iz1-1])) \
        + gz[ 1] * (hy[ 0] * (gx[ 0] * ey[ix1-1,iy2-1,iz1  ] \
        +                     gx[ 1] * ey[ix1  ,iy2-1,iz1  ] \
        +                     gx[ 2] * ey[ix1+1,iy2-1,iz1  ]) \
        +           hy[ 1] * (gx[ 0] * ey[ix1-1,iy2  ,iz1  ] \
        +                     gx[ 1] * ey[ix1  ,iy2  ,iz1  ] \
        +                     gx[ 2] * ey[ix1+1,iy2  ,iz1  ]) \
        +           hy[ 2] * (gx[ 0] * ey[ix1-1,iy2+1,iz1  ] \
        +                     gx[ 1] * ey[ix1  ,iy2+1,iz1  ] \
        +                     gx[ 2] * ey[ix1+1,iy2+1,iz1  ])) \
        + gz[ 2] * (hy[ 0] * (gx[ 0] * ey[ix1-1,iy2-1,iz1+1] \
        +                     gx[ 1] * ey[ix1  ,iy2-1,iz1+1] \
        +                     gx[ 2] * ey[ix1+1,iy2-1,iz1+1]) \
        +           hy[ 1] * (gx[ 0] * ey[ix1-1,iy2  ,iz1+1] \
        +                     gx[ 1] * ey[ix1  ,iy2  ,iz1+1] \
        +                     gx[ 2] * ey[ix1+1,iy2  ,iz1+1]) \
        +           hy[ 2] * (gx[ 0] * ey[ix1-1,iy2+1,iz1+1] \
        +                     gx[ 1] * ey[ix1  ,iy2+1,iz1+1] \
        +                     gx[ 2] * ey[ix1+1,iy2+1,iz1+1]))
    return ey_part


@njit
def interp_ez_3d(ez, gx, gy, hz, ix1, iy1, iz2):
    ez_part = \
          hz[ 0] * (gy[ 0] * (gx[ 0] * ez[ix1-1,iy1-1,iz2-1] \
        +                     gx[ 1] * ez[ix1  ,iy1-1,iz2-1] \
        +                     gx[ 2] * ez[ix1+1,iy1-1,iz2-1]) \
        +           gy[ 1] * (gx[ 0] * ez[ix1-1,iy1  ,iz2-1] \
        +                     gx[ 1] * ez[ix1  ,iy1  ,iz2-1] \
        +                     gx[ 2] * ez[ix1+1,iy1  ,iz2-1]) \
        +           gy[ 2] * (gx[ 0] * ez[ix1-1,iy1+1,iz2-1] \
        +                     gx[ 1] * ez[ix1  ,iy1+1,iz2-1] \
        +                     gx[ 2] * ez[ix1+1,iy1+1,iz2-1])) \
        + hz[ 1] * (gy[ 0] * (gx[ 0] * ez[ix1-1,iy1-1,iz2  ] \
        +                     gx[ 1] * ez[ix1  ,iy1-1,iz2  ] \
        +                     gx[ 2] * ez[ix1+1,iy1-1,iz2  ]) \
        +           gy[ 1] * (gx[ 0] * ez[ix1-1,iy1  ,iz2  ] \
        +                     gx[ 1] * ez[ix1  ,iy1  ,iz2  ] \
        +                     gx[ 2] * ez[ix1+1,iy1  ,iz2  ]) \
        +           gy[ 2] * (gx[ 0] * ez[ix1-1,iy1+1,iz2  ] \
        +                     gx[ 1] * ez[ix1  ,iy1+1,iz2  ] \
        +                     gx[ 2] * ez[ix1+1,iy1+1,iz2  ])) \
        + hz[ 2] * (gy[ 0] * (gx[ 0] * ez[ix1-1,iy1-1,iz2+1] \
        +                     gx[ 1] * ez[ix1  ,iy1-1,iz2+1] \
        +                     gx[ 2] * ez[ix1+1,iy1-1,iz2+1]) \
        +           gy[ 1] * (gx[ 0] * ez[ix1-1,iy1  ,iz2+1] \
        +                     gx[ 1] * ez[ix1  ,iy1  ,iz2+1] \
        +                     gx[ 2] * ez[ix1+1,iy1  ,iz2+1]) \
        +           gy[ 2] * (gx[ 0] * ez[ix1-1,iy1+1,iz2+1] \
        +                     gx[ 1] * ez[ix1  ,iy1+1,iz2+1] \
        +                     gx[ 2] * ez[ix1+1,iy1+1,iz2+1]))
    return ez_part


@njit
def interp_bx_3d(bx, gx, hy, hz, ix1, iy2, iz2):
    bx_part = \
            hz[-1] * (hy[-1] * (gx[-1] * bx[ix1-1,iy2-1,iz2-1] \
        +                     gx[ 0] * bx[ix1  ,iy2-1,iz2-1] \
        +                     gx[ 1] * bx[ix1+1,iy2-1,iz2-1]) \
        +           hy[ 0] * (gx[-1] * bx[ix1-1,iy2  ,iz2-1] \
        +                     gx[ 0] * bx[ix1  ,iy2  ,iz2-1] \
        +                     gx[ 1] * bx[ix1+1,iy2  ,iz2-1]) \
        +           hy[ 1] * (gx[-1] * bx[ix1-1,iy2+1,iz2-1] \
        +                     gx[ 0] * bx[ix1  ,iy2+1,iz2-1] \
        +                     gx[ 1] * bx[ix1+1,iy2+1,iz2-1])) \
        + hz[ 0] * (hy[-1] * (gx[-1] * bx[ix1-1,iy2-1,iz2  ] \
        +                     gx[ 0] * bx[ix1  ,iy2-1,iz2  ] \
        +                     gx[ 1] * bx[ix1+1,iy2-1,iz2  ]) \
        +           hy[ 0] * (gx[-1] * bx[ix1-1,iy2  ,iz2  ] \
        +                     gx[ 0] * bx[ix1  ,iy2  ,iz2  ] \
        +                     gx[ 1] * bx[ix1+1,iy2  ,iz2  ]) \
        +           hy[ 1] * (gx[-1] * bx[ix1-1,iy2+1,iz2  ] \
        +                     gx[ 0] * bx[ix1  ,iy2+1,iz2  ] \
        +                     gx[ 1] * bx[ix1+1,iy2+1,iz2  ])) \
        + hz[ 1] * (hy[-1] * (gx[-1] * bx[ix1-1,iy2-1,iz2+1] \
        +                     gx[ 0] * bx[ix1  ,iy2-1,iz2+1] \
        +                     gx[ 1] * bx[ix1+1,iy2-1,iz2+1]) \
        +           hy[ 0] * (gx[-1] * bx[ix1-1,iy2  ,iz2+1] \
        +                     gx[ 0] * bx[ix1  ,iy2  ,iz2+1] \
        +                     gx[ 1] * bx[ix1+1,iy2  ,iz2+1]) \
        +           hy[ 1] * (gx[-1] * bx[ix1-1,iy2+1,iz2+1] \
        +                     gx[ 0] * bx[ix1  ,iy2+1,iz2+1] \
        +                     gx[ 1] * bx[ix1+1,iy2+1,iz2+1]))
    return bx_part

@njit
def interp_by_3d(by, hx, gy, hz, ix2, iy1, iz2):
    by_part = \
            hz[-1] * (gy[-1] * (hx[-1] * by[ix2-1,iy1-1,iz2-1] \
        +                     hx[ 0] * by[ix2  ,iy1-1,iz2-1] \
        +                     hx[ 1] * by[ix2+1,iy1-1,iz2-1]) \
        +           gy[ 0] * (hx[-1] * by[ix2-1,iy1  ,iz2-1] \
        +                     hx[ 0] * by[ix2  ,iy1  ,iz2-1] \
        +                     hx[ 1] * by[ix2+1,iy1  ,iz2-1]) \
        +           gy[ 1] * (hx[-1] * by[ix2-1,iy1+1,iz2-1] \
        +                     hx[ 0] * by[ix2  ,iy1+1,iz2-1] \
        +                     hx[ 1] * by[ix2+1,iy1+1,iz2-1])) \
        + hz[ 0] * (gy[-1] * (hx[-1] * by[ix2-1,iy1-1,iz2  ] \
        +                     hx[ 0] * by[ix2  ,iy1-1,iz2  ] \
        +                     hx[ 1] * by[ix2+1,iy1-1,iz2  ]) \
        +           gy[ 0] * (hx[-1] * by[ix2-1,iy1  ,iz2  ] \
        +                     hx[ 0] * by[ix2  ,iy1  ,iz2  ] \
        +                     hx[ 1] * by[ix2+1,iy1  ,iz2  ]) \
        +           gy[ 1] * (hx[-1] * by[ix2-1,iy1+1,iz2  ] \
        +                     hx[ 0] * by[ix2  ,iy1+1,iz2  ] \
        +                     hx[ 1] * by[ix2+1,iy1+1,iz2  ])) \
        + hz[ 1] * (gy[-1] * (hx[-1] * by[ix2-1,iy1-1,iz2+1] \
        +                     hx[ 0] * by[ix2  ,iy1-1,iz2+1] \
        +                     hx[ 1] * by[ix2+1,iy1-1,iz2+1]) \
        +           gy[ 0] * (hx[-1] * by[ix2-1,iy1  ,iz2+1] \
        +                     hx[ 0] * by[ix2  ,iy1  ,iz2+1] \
        +                     hx[ 1] * by[ix2+1,iy1  ,iz2+1]) \
        +           gy[ 1] * (hx[-1] * by[ix2-1,iy1+1,iz2+1] \
        +                     hx[ 0] * by[ix2  ,iy1+1,iz2+1] \
        +                     hx[ 1] * by[ix2+1,iy1+1,iz2+1]))
    return by_part

@njit
def interp_bz_3d(bz, hx, hy, gz, ix2, iy2, iz1):
    bz_part = \
            gz[-1] * (hy[-1] * (hx[-1] * bz[ix2-1,iy2-1,iz1-1] \
        +                     hx[ 0] * bz[ix2  ,iy2-1,iz1-1] \
        +                     hx[ 1] * bz[ix2+1,iy2-1,iz1-1]) \
        +           hy[ 0] * (hx[-1] * bz[ix2-1,iy2  ,iz1-1] \
        +                     hx[ 0] * bz[ix2  ,iy2  ,iz1-1] \
        +                     hx[ 1] * bz[ix2+1,iy2  ,iz1-1]) \
        +           hy[ 1] * (hx[-1] * bz[ix2-1,iy2+1,iz1-1] \
        +                     hx[ 0] * bz[ix2  ,iy2+1,iz1-1] \
        +                     hx[ 1] * bz[ix2+1,iy2+1,iz1-1])) \
        + gz[ 0] * (hy[-1] * (hx[-1] * bz[ix2-1,iy2-1,iz1  ] \
        +                     hx[ 0] * bz[ix2  ,iy2-1,iz1  ] \
        +                     hx[ 1] * bz[ix2+1,iy2-1,iz1  ]) \
        +           hy[ 0] * (hx[-1] * bz[ix2-1,iy2  ,iz1  ] \
        +                     hx[ 0] * bz[ix2  ,iy2  ,iz1  ] \
        +                     hx[ 1] * bz[ix2+1,iy2  ,iz1  ]) \
        +           hy[ 1] * (hx[-1] * bz[ix2-1,iy2+1,iz1  ] \
        +                     hx[ 0] * bz[ix2  ,iy2+1,iz1  ] \
        +                     hx[ 1] * bz[ix2+1,iy2+1,iz1  ])) \
        + gz[ 1] * (hy[-1] * (hx[-1] * bz[ix2-1,iy2-1,iz1+1] \
        +                     hx[ 0] * bz[ix2  ,iy2-1,iz1+1] \
        +                     hx[ 1] * bz[ix2+1,iy2-1,iz1+1]) \
        +           hy[ 0] * (hx[-1] * bz[ix2-1,iy2  ,iz1+1] \
        +                     hx[ 0] * bz[ix2  ,iy2  ,iz1+1] \
        +                     hx[ 1] * bz[ix2+1,iy2  ,iz1+1]) \
        +           hy[ 1] * (hx[-1] * bz[ix2-1,iy2+1,iz1+1] \
        +                     hx[ 0] * bz[ix2  ,iy2+1,iz1+1] \
        +                     hx[ 1] * bz[ix2+1,iy2+1,iz1+1]))
    return bz_part

@njit
def em_to_particle_3d(ex, ey, ez, bx, by, bz, x, y, z, dx, dy, dz):
    x_over_dx = x / dx
    y_over_dy = y / dy
    z_over_dz = z / dz

    ix1 = int(np.floor(x_over_dx+0.5))
    gx = get_gx(ix1 - x_over_dx)

    ix2 = int(np.floor(x_over_dx))
    hx = get_gx(ix2 - x_over_dx + 0.5)


    iy1 = int(np.floor(y_over_dy+0.5))
    gy = get_gx(iy1 - y_over_dy)

    iy2 = int(np.floor(y_over_dy))
    hy = get_gx(iy2 - y_over_dy + 0.5)

    iz1 = int(np.floor(z_over_dz+0.5))
    gz = get_gx(iz1 - z_over_dz)

    iz2 = int(np.floor(z_over_dz))
    hz = get_gx(iz2 - z_over_dz + 0.5)

    ex_part = interp_ex_3d(ex, hx, gy, gz, ix2, iy1, iz1)
    ey_part = interp_ey_3d(ey, gx, hy, gz, ix1, iy2, iz1)
    ez_part = interp_ez_3d(ez, gx, gy, hz, ix1, iy1, iz2)

    bx_part = interp_bx_3d(bx, gx, hy, hz, ix1, iy2, iz2)
    by_part = interp_by_3d(by, hx, gy, hz, ix2, iy1, iz2)
    bz_part = interp_bz_3d(bz, hx, hy, gz, ix2, iy2, iz1)

    return ex_part, ey_part, ez_part, bx_part, by_part, bz_part

@njit
def get_gx(delta):
    delta2 = delta*delta
    return [
        0.5*(0.25 + delta2 + delta),
        0.75 - delta2,
        0.5*(0.25 + delta2 - delta),
    ]
