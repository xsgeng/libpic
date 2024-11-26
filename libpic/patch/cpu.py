import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

""" Parallel functions for patches """
@njit(cache=True, parallel=True)
def sync_currents(
    jx_list, jy_list, jz_list, rho_list,
    xmin_index_list,
    xmax_index_list,
    ymin_index_list,
    ymax_index_list,
    npatches, nx, ny, ng
):
    all_fields = [jx_list, jy_list, jz_list, rho_list]
    for i in prange(npatches*4):
        field = all_fields[i%4]
        ipatch = i//4
        xmin_index = xmin_index_list[ipatch]
        xmax_index = xmax_index_list[ipatch]
        ymin_index = ymin_index_list[ipatch]
        ymax_index = ymax_index_list[ipatch]
        if xmin_index >= 0:
            field[ipatch][:ng, :ny] += field[xmin_index][nx:nx+ng, :ny]
            field[xmin_index][nx:nx+ng, :ny] = 0
        if ymin_index >= 0:
            field[ipatch][:nx, :ng] += field[ymin_index][:nx, ny:ny+ng]
            field[ymin_index][:nx, ny:ny+ng] = 0
        if xmax_index >= 0:
            field[ipatch][nx-ng:nx, :ny] += field[xmax_index][-ng:, :ny]
            field[xmax_index][-ng:, :ny] = 0
        if ymax_index >= 0:
            field[ipatch][:nx, ny-ng:ny] += field[ymax_index][:nx, -ng:]
            field[ymax_index][:nx, -ng:] = 0

        # corners
        if ymin_index >= 0:
            xminymin_index = xmin_index_list[ymin_index]
            if xminymin_index >= 0:
                field[ipatch][:ng, :ng] += field[xminymin_index][nx:nx+ng, ny:ny+ng]
                field[xminymin_index][nx:nx+ng, ny:ny+ng] = 0
            xmaxymin_index = xmax_index_list[ymin_index]
            if xmaxymin_index >= 0:
                field[ipatch][nx-ng:nx, :ng] += field[xmaxymin_index][-ng:, ny:ny+ng]
                field[xmaxymin_index][-ng:, ny:ny+ng] = 0
        if ymax_index >= 0:
            xminymax_index = xmin_index_list[ymax_index]
            if xminymax_index >= 0:
                field[ipatch][:ng, ny-ng:ny] += field[xminymax_index][nx:nx+ng, -ng:]
                field[xminymax_index][nx:nx+ng, -ng:] = 0
            xmaxymax_index = xmax_index_list[ymax_index]
            if xmaxymax_index >= 0:
                field[ipatch][nx-ng:nx, ny-ng:ny] += field[xmaxymax_index][-ng:, -ng:]
                field[xmaxymax_index][-ng:, -ng:] = 0


@njit(cache=True, parallel=True)
def sync_guard_fields(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    xmin_index_list,
    xmax_index_list,
    ymin_index_list,
    ymax_index_list,
    npatches, nx, ny, ng
):
    all_fields = [ex_list, ey_list, ez_list, bx_list, by_list, bz_list]
    for i in prange(npatches*6):
        field = all_fields[i%6]
        ipatch = i//6
        xmin_index = xmin_index_list[ipatch]
        xmax_index = xmax_index_list[ipatch]
        ymin_index = ymin_index_list[ipatch]
        ymax_index = ymax_index_list[ipatch]
        if xmin_index >= 0:
            field[ipatch][-ng:, :ny] = field[xmin_index][nx-ng:nx, :ny]
        if ymin_index >= 0:
            field[ipatch][:nx, -ng:] = field[ymin_index][:nx, ny-ng:ny]
        if xmax_index >= 0:
            field[ipatch][nx:nx+ng, :ny] = field[xmax_index][:ng, :ny]
        if ymax_index >= 0:
            field[ipatch][:nx, ny:ny+ng] = field[ymax_index][:nx, :ng]

        # corners
        if ymin_index >= 0:
            xminymin_index = xmin_index_list[ymin_index]
            if xminymin_index >= 0:
                field[ipatch][-ng:, -ng:] = field[xminymin_index][nx-ng:nx, ny-ng:ny]
            xmaxymin_index = xmax_index_list[ymin_index]
            if xmaxymin_index >= 0:
                field[ipatch][nx:nx+ng, -ng:] = field[xmaxymin_index][:ng, ny-ng:ny]
        if ymax_index >= 0:
            xminymax_index = xmin_index_list[ymax_index]
            if xminymax_index >= 0:
                field[ipatch][-ng:, ny:ny+ng] = field[xminymax_index][nx-ng:nx, :ng]
            xmaxymax_index = xmax_index_list[ymax_index]
            if xmaxymax_index >= 0:
                field[ipatch][nx:nx+ng, ny:ny+ng] = field[xmaxymax_index][:ng, :ng]


@njit(cache=False, parallel=True)
def get_num_macro_particles(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc) -> NDArray[np.int64]:
    num_particles = np.zeros(npatches, dtype=np.int64)
    for ipatch in prange(npatches):
        xaxis =  xaxis_list[ipatch]
        yaxis =  yaxis_list[ipatch]

        for x_grid in xaxis:
            for y_grid in yaxis:
                dens = density_func(x_grid, y_grid)
                if dens > dens_min:
                    num_particles[ipatch] += ppc
    return num_particles


@njit(cache=False, parallel=True)
def fill_particles(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc, x_list, y_list, w_list):
    dx = xaxis_list[0][1] - xaxis_list[0][0]
    dy = yaxis_list[0][1] - yaxis_list[0][0]
    for ipatch in prange(npatches):
        xaxis =  xaxis_list[ipatch]
        yaxis =  yaxis_list[ipatch]
        x = x_list[ipatch]
        y = y_list[ipatch]
        w = w_list[ipatch]
        ipart = 0
        for x_grid in xaxis:
            for y_grid in yaxis:
                dens = density_func(x_grid, y_grid)
                if dens > dens_min:
                    x[ipart:ipart+ppc] = np.random.uniform(-dx/2, dx/2) + x_grid
                    y[ipart:ipart+ppc] = np.random.uniform(-dy/2, dy/2) + y_grid
                    w[ipart:ipart+ppc] = dens*dx*dy / ppc
                    ipart += ppc
