import numpy as np
from libpic.deposition_2d import current_deposit_2d
from libpic.interpolation_2d import interpolation_2d


from numba import njit, prange

from libpic.maxwell_2d import update_bfield_2d, update_efield_2d
from libpic.pusher import boris_cpu, push_position_2d


""" Parallel functions for patches """
@njit(parallel=True)
def interpolation(
    x_list, y_list,
    ex_part_list, ey_part_list, ez_part_list,
    bx_part_list, by_part_list, bz_part_list,
    npart_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    xaxis_list, yaxis_list,
    npatches,
    dx, dy,
    pruned_list,
) -> None:
    for ipatch in prange(npatches):
        x = x_list[ipatch]
        y = y_list[ipatch]
        ex_part = ex_part_list[ipatch]
        ey_part = ey_part_list[ipatch]
        ez_part = ez_part_list[ipatch]
        bx_part = bx_part_list[ipatch]
        by_part = by_part_list[ipatch]
        bz_part = bz_part_list[ipatch]
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        x0 = xaxis_list[ipatch][0]
        y0 = yaxis_list[ipatch][0]
        x = x_list[ipatch]
        y = y_list[ipatch]
        pruned = pruned_list[ipatch]
        npart = len(pruned)
        interpolation_2d(
            x, y, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, npart,
            ex, ey, ez, bx, by, bz,
            dx, dy, x0, y0,
            pruned,
        )


@njit(parallel=True)
def current_deposition(
    rho_list,
    jx_list, jy_list, jz_list,
    xaxis_list, yaxis_list,
    x_list, y_list, ux_list, uy_list, uz_list,
    inv_gamma_list,
    pruned_list,
    npatches,
    dx, dy, dt, w_list, q,
) -> None:
    for ipatch in prange(npatches):
        rho = rho_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]
        x0 = xaxis_list[ipatch][0]
        y0 = yaxis_list[ipatch][0]
        x = x_list[ipatch]
        y = y_list[ipatch]
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        w = w_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]
        pruned = pruned_list[ipatch]
        npart = len(pruned)

        jx[:] = 0
        jy[:] = 0
        jz[:] = 0
        rho[:] = 0
        current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)


@njit(parallel=True)
def update_efield_patches(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]

        update_efield_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, dx, dy, dt, nx, ny, n_guard)


@njit(parallel=True)
def update_bfield_patches(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        update_bfield_2d(ex, ey, ez, bx, by, bz, dx, dy, dt, nx, ny, n_guard)


@njit(parallel=True)
def boris_push(
    ux_list, uy_list, uz_list, inv_gamma_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches, q, npart_list, pruned_list, dt
) -> None:
    for ipatch in prange(npatches):
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]

        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]

        pruned = pruned_list[ipatch]
        npart = len(pruned)
        for ipart in range(npart):
            if not pruned[ipart]:
                ux[ipart], uy[ipart], uz[ipart], inv_gamma[ipart] = boris_cpu(
                    ux[ipart], uy[ipart], uz[ipart],
                    ex[ipart], ey[ipart], ez[ipart],
                    bx[ipart], by[ipart], bz[ipart],
                    q, dt
                )


@njit(parallel=True)
def push_position(
    x_list, y_list,
    ux_list, uy_list, inv_gamma_list,
    npatches, pruned_list,
    dt,
) -> None:
    for ipatch in prange(npatches):
        x = x_list[ipatch]
        y = y_list[ipatch]

        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]

        pruned = pruned_list[ipatch]
        npart = len(pruned)
        push_position_2d( x, y, ux, uy, inv_gamma, npart, pruned, dt )


@njit(parallel=True)
def sync_currents(
    jx_list, jy_list, jz_list,
    xmin_index_list,
    xmax_index_list,
    ymin_index_list,
    ymax_index_list,
    npatches, nx, ny, ng
):
    all_fields = [jx_list, jy_list, jz_list]
    for i in prange(npatches*3):
        field = all_fields[i%3]
        ipatch = i//3
        xmin_index = xmin_index_list[ipatch]
        xmax_index = xmax_index_list[ipatch]
        ymin_index = ymin_index_list[ipatch]
        ymax_index = ymax_index_list[ipatch]
        if xmin_index >= 0:
            field[ipatch][:ng, :ny] += field[xmin_index][nx:nx+ng, :ny]
        if ymin_index >= 0:
            field[ipatch][:nx, :ng] = field[ymin_index][:nx, ny:ny+ng]
        if xmax_index >= 0:
            field[ipatch][nx-ng:nx, :ny] += field[xmax_index][-ng:, :ny]
        if ymax_index >= 0:
            field[ipatch][:nx, ny-ng:ny] = field[ymax_index][:nx, -ng:]

@njit(parallel=True)
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
    all_fields = [ex_list, ey_list, ez_list, bx_list, by_list, bz_list, jx_list, jy_list, jz_list]
    for i in prange(npatches*9):
        field = all_fields[i%9]
        ipatch = i//9
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


@njit(parallel=True, cache=True)
def mark_out_of_bound_as_pruned(
    x_list, y_list,
    npart_list,
    pruned_list,
    xaxis_list,
    yaxis_list,
    npatches, dx, dy,
):
    for ipatches in prange(npatches):
        x = x_list[ipatches]
        y = y_list[ipatches]
        pruned = pruned_list[ipatches]
        xaxis = xaxis_list[ipatches]
        yaxis = yaxis_list[ipatches]

        xmin = xaxis[ 0] - 0.5*dx
        xmax = xaxis[-1] + 0.5*dx
        ymin = yaxis[ 0] - 0.5*dy
        ymax = yaxis[-1] + 0.5*dy
        # mark pruned
        for i in range(len(pruned)):
            if pruned[i]:
                x[i] = np.nan
                y[i] = np.nan
                continue
            if x[i] < xmin:
                pruned[i] = True
                continue
            if x[i] > xmax:
                pruned[i] = True
                continue
            if y[i] < ymin:
                pruned[i] = True
                continue
            if y[i] > ymax:
                pruned[i] = True
                continue


@njit
def count_new_particles(
    x_list, y_list,
    xmin_index, xmax_index, ymin_index, ymax_index,
    xaxis_list, yaxis_list,
    dx, dy,
):
    npart_out_of_bound = 0

    if xmin_index >= 0:
        xmax = xaxis_list[xmin_index][-1] + 0.5*dx
        x = x_list[xmin_index]
        npart = len(x)
        for i in range(npart):
            if x[i] > xmax:
                npart_out_of_bound += 1

    if xmax_index >= 0:
        xmin = xaxis_list[xmax_index][0] - 0.5*dx
        x = x_list[xmax_index]
        npart = len(x)
        for i in range(npart):
            if x[i] < xmin:
                npart_out_of_bound += 1

    if ymin_index >= 0:
        ymax = yaxis_list[ymin_index][-1] + 0.5*dy
        y = y_list[ymin_index]
        npart = len(y)
        for i in range(npart):
            if y[i] > ymax:
                npart_out_of_bound += 1

    if ymax_index >= 0:
        ymin = yaxis_list[ymax_index][0] - 0.5*dy
        y = y_list[ymax_index]
        npart = len(y)
        for i in range(npart):
            if y[i] < ymin:
                npart_out_of_bound += 1

    return npart_out_of_bound


@njit(parallel=True, cache=True)
def get_npart_to_extend(
    x_list, y_list,
    npart_list,
    pruned_list,
    xaxis_list,
    yaxis_list,
    xmin_index_list,
    xmax_index_list,
    ymin_index_list,
    ymax_index_list,
    npatches, dx, dy,
):
    """ 
    count the number of particles to be extended, and return the number of new particles
    """
    npart_to_extend = np.zeros(npatches, dtype='i8')
    npart_new = np.zeros(npatches, dtype='i8')
    for ipatches in prange(npatches):
        pruned = pruned_list[ipatches]

        xmin_index = xmin_index_list[ipatches]
        xmax_index = xmax_index_list[ipatches]
        ymin_index = ymin_index_list[ipatches]
        ymax_index = ymax_index_list[ipatches]

        # 0, 1 for x and y
        npart_new_ = count_new_particles(x_list, y_list,
                                        xmin_index, xmax_index, ymin_index, ymax_index, xaxis_list, yaxis_list, dx, dy)

        # count vacants
        npruned = 0
        for pruned_ in pruned:
            if pruned_: npruned += 1

        if npart_new_ - npruned > 0:
            # reserved more space for new particles in the following loops
            npart_to_extend[ipatches] = npart_new_ - npruned + len(pruned)*0.25
        # else:
        #     npart_to_extend[ipatches] = 0

        npart_new[ipatches] = npart_new_
    return npart_to_extend, npart_new


@njit(inline="always")
def fill_boundary_particles_to_buffer(
    buffer,
    npart_list, xaxis_list, yaxis_list,
    xmin_index, xmax_index, ymin_index, ymax_index,
    dx, dy,
    attrs_list,
):
    npart_new = buffer.shape[0]
    ibuff = 0
    # on xmin boundary
    if xmin_index >= 0:
        x_on_xmin = attrs_list[0][xmin_index]
        xmax = xaxis_list[xmin_index][-1] + 0.5*dx
        for ipart in range(npart_list[xmin_index]):
            if ibuff >= npart_new:
                break
            if x_on_xmin[ipart] > xmax:
                for iattr, attr in enumerate(attrs_list):
                    buffer[ibuff, iattr] = attr[xmin_index][ipart]
                ibuff += 1

    # on xmax boundary
    if xmax_index >= 0:
        x_on_xmax = attrs_list[0][xmax_index]
        xmin = xaxis_list[xmax_index][ 0] - 0.5*dx
        for ipart in range(npart_list[xmax_index]):
            if ibuff >= npart_new:
                break
            if x_on_xmax[ipart] < xmin:
                for iattr, attr in enumerate(attrs_list):
                    buffer[ibuff, iattr] = attr[xmax_index][ipart]
                ibuff += 1

    # on ymin boundary
    if ymin_index >= 0:
        y_on_ymin = attrs_list[1][ymin_index]
        ymax = yaxis_list[ymin_index][-1] + 0.5*dy
        for ipart in range(npart_list[ymin_index]):
            if ibuff >= npart_new:
                break
            if y_on_ymin[ipart] > ymax:
                for iattr, attr in enumerate(attrs_list):
                    buffer[ibuff, iattr] = attr[ymin_index][ipart]
                ibuff += 1

    # on ymax boundary
    if ymax_index >= 0:
        y_on_ymax = attrs_list[1][ymax_index]
        ymin = yaxis_list[ymax_index][ 0] - 0.5*dy
        for ipart in range(npart_list[ymax_index]):
            if ibuff >= npart_new:
                break
            if y_on_ymax[ipart] < ymin:
                for iattr, attr in enumerate(attrs_list):
                    buffer[ibuff, iattr] = attr[ymax_index][ipart]
                ibuff += 1

    assert ibuff == npart_new


@njit
def fill_particles_from_boundary(
    npart_list,
    pruned_list,
    xaxis_list,
    yaxis_list,
    xmin_index_list,
    xmax_index_list,
    ymin_index_list,
    ymax_index_list,
    npart_new_list,
    npatches, dx, dy,
    *attrs_list,
):
    nattrs = len(attrs_list)
    for ipatches in prange(npatches):
        npart_new = npart_new_list[ipatches]
        if npart_new <= 0:
            continue

        pruned = pruned_list[ipatches]

        xmin_index = xmin_index_list[ipatches]
        xmax_index = xmax_index_list[ipatches]
        ymin_index = ymin_index_list[ipatches]
        ymax_index = ymax_index_list[ipatches]

        buffer = np.zeros((npart_new, nattrs))
        fill_boundary_particles_to_buffer(buffer, npart_list, xaxis_list, yaxis_list,
                                          xmin_index, xmax_index, ymin_index, ymax_index,
                                          dx, dy, attrs_list)
        # if ipatches == 0:
        #     print(buffer[:, 0], len(pruned), npart_new)
        # fill the pruned
        ibuff = 0
        for ipart in range(len(pruned)):
            if ibuff >= npart_new:
                break
            if pruned[ipart]:
                for iattr in range(nattrs):
                    attrs_list[iattr][ipatches][ipart] = buffer[ibuff, iattr]
                    # if iattr == 0 and ipatches == 0:
                    #     print(attrs_list[iattr][ipatches][ipart], buffer[ibuff, iattr])
                pruned[ipart] = False
                ibuff += 1


@njit(parallel=True)
def get_num_macro_particles(density_func, xaxis_list, yaxis_list, npatches, dens_min, ppc) -> np.ndarray:
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


@njit(parallel=True)
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
                    w[ipart:ipart+ppc] = dens / ppc
                    ipart += ppc