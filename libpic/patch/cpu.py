import numpy as np
from libpic.deposition_2d import current_deposit_2d
from libpic.interpolation_2d import interpolation_2d


from numba import njit, prange, get_num_threads, get_thread_id, typed

from libpic.maxwell_2d import update_bfield_2d, update_efield_2d
from libpic.pusher import boris, boris_cpu, push_position_2d


""" Parallel functions for patches """
@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
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

        current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
def boris_push(
    ux_list, uy_list, uz_list, inv_gamma_list,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    npatches, q, m, npart_list, pruned_list, dt
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
        boris( ux, uy, uz, inv_gamma, ex, ey, ez, bx, by, bz, q, m, npart, pruned, dt )


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
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
            field[ipatch][:nx, :ng] += field[ymin_index][:nx, ny:ny+ng]
        if xmax_index >= 0:
            field[ipatch][nx-ng:nx, :ny] += field[xmax_index][-ng:, :ny]
        if ymax_index >= 0:
            field[ipatch][:nx, ny-ng:ny] += field[ymax_index][:nx, -ng:]

        # corners
        if ymin_index >= 0:
            xminymin_index = xmin_index_list[ymin_index]
            if xminymin_index >= 0:
                field[ipatch][:ng, :ng] += field[xminymin_index][nx:nx+ng, ny:ny+ng]
            xmaxymin_index = xmax_index_list[ymin_index]
            if xmaxymin_index >= 0:
                field[ipatch][nx-ng:nx, :ng] += field[xmaxymin_index][-ng:, ny:ny+ng]
        if ymax_index >= 0:
            xminymax_index = xmin_index_list[ymax_index]
            if xminymax_index >= 0:
                field[ipatch][:ng, ny-ng:ny] += field[xminymax_index][nx:nx+ng, -ng:]
            xmaxymax_index = xmax_index_list[ymax_index]
            if xmaxymax_index >= 0:
                field[ipatch][nx-ng:nx, ny-ng:ny] += field[xmaxymax_index][-ng:, -ng:]


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
                x[i] = np.nan
                y[i] = np.nan
                continue
            if x[i] > xmax:
                pruned[i] = True
                x[i] = np.nan
                y[i] = np.nan
                continue
            if y[i] < ymin:
                pruned[i] = True
                x[i] = np.nan
                y[i] = np.nan
                continue
            if y[i] > ymax:
                pruned[i] = True
                x[i] = np.nan
                y[i] = np.nan
                continue

@njit
def count_outgoing_particles(x, y, xmin, xmax, ymin, ymax):
    npart_xmin = 0
    npart_xmax = 0
    npart_ymin = 0
    npart_ymax = 0
    npart_xminymin = 0
    npart_xmaxymin = 0
    npart_xminymax = 0
    npart_xmaxymax = 0
    for ip in range(len(x)):
        if y[ip] < ymin:
            if x[ip] < xmin:
                npart_xminymin += 1
                continue
            elif x[ip] > xmax:
                npart_xmaxymin += 1
                continue
            else:
                npart_ymin += 1
                continue
        elif y[ip] > ymax:
            if x[ip] < xmin:
                npart_xminymax += 1
                continue
            elif x[ip] > xmax:
                npart_xmaxymax += 1
                continue
            else:
                npart_ymax += 1
                continue
        else:
            if x[ip] < xmin:
                npart_xmin += 1
                continue
            elif x[ip] > xmax:
                npart_xmax += 1
                continue
    return (npart_xmin, npart_xmax, npart_ymin, npart_ymax,
            npart_xminymin, npart_xmaxymin, npart_xminymax, npart_xmaxymax)


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
    npart_incoming = np.zeros(npatches, dtype='i8')
    npart_outgoing = np.zeros((8, npatches), dtype='i8')
    for ipatches in prange(npatches):
        x = x_list[ipatches]
        y = y_list[ipatches]
        xmin = xaxis_list[ipatches][ 0] - 0.5*dx
        xmax = xaxis_list[ipatches][-1] + 0.5*dx
        ymin = yaxis_list[ipatches][ 0] - 0.5*dy
        ymax = yaxis_list[ipatches][-1] + 0.5*dy
        npart_xmin, npart_xmax, npart_ymin, npart_ymax,\
        npart_xminymin, npart_xmaxymin, npart_xminymax, npart_xmaxymax = count_outgoing_particles(x, y, xmin, xmax, ymin, ymax)
        npart_outgoing[0, ipatches] = npart_xmin
        npart_outgoing[1, ipatches] = npart_xmax
        npart_outgoing[2, ipatches] = npart_ymin
        npart_outgoing[3, ipatches] = npart_ymax
        npart_outgoing[4, ipatches] = npart_xminymin
        npart_outgoing[5, ipatches] = npart_xmaxymin
        npart_outgoing[6, ipatches] = npart_xminymax
        npart_outgoing[7, ipatches] = npart_xmaxymax

    for ipatches in prange(npatches):
        pruned = pruned_list[ipatches]

        xmin_index = xmin_index_list[ipatches]
        xmax_index = xmax_index_list[ipatches]
        ymin_index = ymin_index_list[ipatches]
        ymax_index = ymax_index_list[ipatches]
        # corners
        xminymin_index = xmin_index_list[ymin_index] if ymin_index >= 0 else -1
        xmaxymin_index = xmax_index_list[ymin_index] if ymin_index >= 0 else -1
        xminymax_index = xmin_index_list[ymax_index] if ymax_index >= 0 else -1
        xmaxymax_index = xmax_index_list[ymax_index] if ymax_index >= 0 else -1

        npart_new = 0
        if xmax_index >= 0:
            npart_new += npart_outgoing[0, xmax_index]
        if xmin_index >= 0:
            npart_new += npart_outgoing[1, xmin_index]
        if ymax_index >= 0:
            npart_new += npart_outgoing[2, ymax_index]
        if ymin_index >= 0:
            npart_new += npart_outgoing[3, ymin_index]
        # corners
        if xmaxymax_index >= 0:
            npart_new += npart_outgoing[4, xmaxymax_index]
        if xminymax_index >= 0:
            npart_new += npart_outgoing[5, xminymax_index]
        if xmaxymin_index >= 0:
            npart_new += npart_outgoing[6, xmaxymin_index]
        if xminymin_index >= 0:
            npart_new += npart_outgoing[7, xminymin_index]

        # count vacants
        npruned = 0
        for pruned_ in pruned:
            if pruned_: npruned += 1

        if (npart_new - npruned) > 0:
            # reserved more space for new particles in the following loops
            npart_to_extend[ipatches] = npart_new - npruned + int(len(pruned)*0.25)

        npart_incoming[ipatches] = npart_new
    return npart_to_extend, npart_incoming, npart_outgoing

@njit
def get_incoming_index(
    x_list, y_list, 
    xmin_index, xmax_index, ymin_index, ymax_index,
    xminymin_index, xmaxymin_index, xminymax_index, xmaxymax_index,
    xaxis_list, yaxis_list,
    dx, dy,
    xmin_indices, xmax_indices, ymin_indices, ymax_indices,
    xminymin_indices, xmaxymin_indices, xminymax_indices, xmaxymax_indices,
):
    """
    fill the buffer with boundary particles

    Parameters
    ----------
    x_list, y_list:
        list of xy coordinates of patches
    *_index:
        indices of patches on the boundaries
    xaxis_list, yaxis_list:
        list of x and y axis of patches
    dx, dy:
        cell sizes
    *_indices:
        indices of incoming particles from xmin, xmax, ymin, ymax, xminymin, xmaxymin, xminymax, xmaxymax
    
    """
    # on xmin boundary
    if xmin_index >= 0:
        x_on_xmin = x_list[xmin_index]
        y_on_xmin = y_list[xmin_index]
        xmax = xaxis_list[xmin_index][-1] + 0.5*dx
        ymin = yaxis_list[xmin_index][ 0] - 0.5*dy
        ymax = yaxis_list[xmin_index][-1] + 0.5*dy
        npart = len(x_on_xmin)

        i = 0
        for ipart in range(npart):
            if (x_on_xmin[ipart] > xmax) and (y_on_xmin[ipart] >= ymin) and (y_on_xmin[ipart] <= ymax):
                xmin_indices[i] = ipart
                i += 1
                if i >= len(xmin_indices):
                    break
    # on xmax boundary
    if xmax_index >= 0:
        x_on_xmax = x_list[xmax_index]
        y_on_xmax = y_list[xmax_index]
        xmin = xaxis_list[xmax_index][ 0] - 0.5*dx
        ymin = yaxis_list[xmax_index][ 0] - 0.5*dy
        ymax = yaxis_list[xmax_index][-1] + 0.5*dy
        npart = len(x_on_xmax)

        i = 0
        for ipart in range(npart):
            if (x_on_xmax[ipart] < xmin) and (y_on_xmax[ipart] >= ymin) and (y_on_xmax[ipart] <= ymax):
                xmax_indices[i] = ipart
                i += 1
                if i >= len(xmax_indices):
                    break
    # on ymin boundary
    if ymin_index >= 0:
        x_on_ymin = x_list[ymin_index]
        y_on_ymin = y_list[ymin_index]
        xmin = xaxis_list[ymin_index][ 0] - 0.5*dx
        xmax = xaxis_list[ymin_index][-1] + 0.5*dx
        ymax = yaxis_list[ymin_index][-1] + 0.5*dy
        npart = len(y_on_ymin)

        i = 0
        for ipart in range(npart):
            if (x_on_ymin[ipart] >= xmin) and (x_on_ymin[ipart] <= xmax) and (y_on_ymin[ipart] > ymax):
                ymin_indices[i] = ipart
                i += 1
                if i >= len(ymin_indices):
                    break
    # on ymax boundary
    if ymax_index >= 0:
        x_on_ymax = x_list[ymax_index]
        y_on_ymax = y_list[ymax_index]
        xmin = xaxis_list[ymax_index][ 0] - 0.5*dx
        xmax = xaxis_list[ymax_index][-1] + 0.5*dx
        ymin = yaxis_list[ymax_index][ 0] - 0.5*dy
        npart = len(y_on_ymax)

        i = 0
        for ipart in range(npart):
            if (x_on_ymax[ipart] >= xmin) and (x_on_ymax[ipart] <= xmax) and (y_on_ymax[ipart] < ymin):
                ymax_indices[i] = ipart
                i += 1
                if i >= len(ymax_indices):
                    break
    # on xminymin boundary
    if xminymin_index >= 0:
        x_on_xminymin = x_list[xminymin_index]
        y_on_xminymin = y_list[xminymin_index]
        xmax = xaxis_list[xminymin_index][-1] + 0.5*dx
        ymax = yaxis_list[xminymin_index][-1] + 0.5*dy
        npart = len(x_on_xminymin)

        i = 0
        for ipart in range(npart):
            if (x_on_xminymin[ipart] > xmax) and (y_on_xminymin[ipart] > ymax):
                xminymin_indices[i] = ipart
                i += 1
                if i >= len(xminymin_indices):
                    break
    # on xmaxymin boundary
    if xmaxymin_index >= 0:
        x_on_xmaxymin = x_list[xmaxymin_index]
        y_on_xmaxymin = y_list[xmaxymin_index]
        xmin = xaxis_list[xmaxymin_index][ 0] - 0.5*dx
        ymax = yaxis_list[xmaxymin_index][-1] + 0.5*dy
        npart = len(x_on_xmaxymin)

        i = 0
        for ipart in range(npart):
            if (x_on_xmaxymin[ipart] < xmin) and (y_on_xmaxymin[ipart] > ymax):
                xmaxymin_indices[i] = ipart
                i += 1
                if i >= len(xmaxymin_indices):
                   break
    # on xminymax boundary
    if xminymax_index >= 0:
        x_on_xminymax = x_list[xminymax_index]
        y_on_xminymax = y_list[xminymax_index]
        xmax = xaxis_list[xminymax_index][-1] + 0.5*dx
        ymin = yaxis_list[xminymax_index][ 0] - 0.5*dy
        npart = len(x_on_xminymax)

        i = 0
        for ipart in range(npart):
            if (x_on_xminymax[ipart] > xmax) and (y_on_xminymax[ipart] < ymin):
                xminymax_indices[i] = ipart
                i += 1
                if i >= len(xminymax_indices):
                   break
    # on xmaxymax boundary
    if xmaxymax_index >= 0:
        x_on_xmaxymax = x_list[xmaxymax_index]
        y_on_xmaxymax = y_list[xmaxymax_index]
        xmin = xaxis_list[xmaxymax_index][ 0] - 0.5*dx
        ymin = yaxis_list[xmaxymax_index][ 0] - 0.5*dy
        npart = len(x_on_xmaxymax)

        i = 0
        for ipart in range(npart):
           if (x_on_xmaxymax[ipart] < xmin) and (y_on_xmaxymax[ipart] < ymin):
                xmaxymax_indices[i] = ipart
                i += 1
                if i >= len(xmaxymax_indices):
                   break

@njit
def fill_boundary_particles_to_buffer(
    attrs_list,
    xmin_indices, xmax_indices, ymin_indices, ymax_indices,
    xminymin_indices, xmaxymin_indices, xminymax_indices, xmaxymax_indices,
    xmin_index, xmax_index, ymin_index, ymax_index,
    xminymin_index, xmaxymin_index, xminymax_index, xmaxymax_index,
    buffer,
):
    """
    fill the buffer with boundary particles

    Parameters
    ----------
    buffer: size of (npart_new, nattrs)
        buffer to be filled
    *_indices:
        indices of incoming particles from xmin, xmax, ymin, ymax
    *_index:
        index of the patch on xmin, xmax, ymin, ymax
    attrs_list: [iattr][ipatch][ipart]
        list of particle attributes
    """
    nattrs = len(attrs_list)
    for iattr in range(nattrs):
        attr_list = attrs_list[iattr]
        ibuff = 0

        attr = attr_list[xmin_index]
        for idx in xmin_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        attr = attr_list[xmax_index]
        for idx in xmax_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        attr = attr_list[ymin_index]
        for idx in ymin_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        attr = attr_list[ymax_index]
        for idx in ymax_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        # corners
        attr = attr_list[xminymin_index]
        for idx in xminymin_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        attr = attr_list[xmaxymin_index]
        for idx in xmaxymin_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        attr = attr_list[xminymax_index]
        for idx in xminymax_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1

        attr = attr_list[xmaxymax_index]
        for idx in xmaxymax_indices:
            buffer[ibuff, iattr] = attr[idx]
            ibuff += 1


@njit(cache=True, parallel=True)
def fill_particles_from_boundary(
    pruned_list,
    xaxis_list,
    yaxis_list,
    xmin_index_list,
    xmax_index_list,
    ymin_index_list,
    ymax_index_list,
    npart_incoming,
    npart_outgoing,
    npatches, dx, dy,
    *attrs_list,
):
    nattrs = len(attrs_list)
    for ipatches in prange(npatches):
        npart_new = npart_incoming[ipatches]
        if npart_new <= 0:
            continue

        pruned = pruned_list[ipatches]

        xmin_index = xmin_index_list[ipatches]
        xmax_index = xmax_index_list[ipatches]
        ymin_index = ymin_index_list[ipatches]
        ymax_index = ymax_index_list[ipatches]
        xminymin_index = ymin_index_list[xmin_index] if xmin_index >= 0 else -1
        xmaxymin_index = ymin_index_list[xmax_index] if xmax_index >= 0 else -1
        xminymax_index = ymax_index_list[xmin_index] if xmin_index >= 0 else -1
        xmaxymax_index = ymax_index_list[xmax_index] if xmax_index >= 0 else -1

        x_list = attrs_list[0]
        y_list = attrs_list[1]
         
        # number of particles coming from xmax boundary is
        # the number of particles going through xmin boundary 
        # in the xmax_index patch. 
        npart_incoming_xmax = npart_outgoing[0, xmax_index] if  xmax_index >= 0 else 0
        npart_incoming_xmin = npart_outgoing[1, xmin_index] if  xmin_index >= 0 else 0
        npart_incoming_ymax = npart_outgoing[2, ymax_index] if  ymax_index >= 0 else 0
        npart_incoming_ymin = npart_outgoing[3, ymin_index] if  ymin_index >= 0 else 0
        # corners
        npart_incoming_xmaxymax = npart_outgoing[4, xmaxymax_index] if xminymin_index >= 0 else 0
        npart_incoming_xminymax = npart_outgoing[5, xminymax_index] if xmaxymin_index >= 0 else 0
        npart_incoming_xmaxymin = npart_outgoing[6, xmaxymin_index] if xminymax_index >= 0 else 0
        npart_incoming_xminymin = npart_outgoing[7, xminymin_index] if xmaxymax_index >= 0 else 0

        # indices of particles coming from boundary
        xmin_incoming_indices = np.zeros(npart_incoming_xmin, dtype='i8')
        xmax_incoming_indices = np.zeros(npart_incoming_xmax, dtype='i8')
        ymin_incoming_indices = np.zeros(npart_incoming_ymin, dtype='i8')
        ymax_incoming_indices = np.zeros(npart_incoming_ymax, dtype='i8')
        xminymin_incoming_indices = np.zeros(npart_incoming_xminymin, dtype='i8')
        xmaxymin_incoming_indices = np.zeros(npart_incoming_xmaxymin, dtype='i8')
        xminymax_incoming_indices = np.zeros(npart_incoming_xminymax, dtype='i8')
        xmaxymax_incoming_indices = np.zeros(npart_incoming_xmaxymax, dtype='i8')

        # assert npart_incoming_xmin + npart_incoming_xmax + npart_incoming_ymin + npart_incoming_ymax == npart_new

        get_incoming_index(
            x_list, y_list, 
            xmin_index, xmax_index, ymin_index, ymax_index,
            xminymin_index, xmaxymin_index, xminymax_index, xmaxymax_index,
            xaxis_list, yaxis_list,
            dx, dy,
            xmin_incoming_indices, xmax_incoming_indices, ymin_incoming_indices, ymax_incoming_indices,
            xminymin_incoming_indices, xmaxymin_incoming_indices, xminymax_incoming_indices, xmaxymax_incoming_indices,
        )

        buffer = np.zeros((npart_new, nattrs))
        fill_boundary_particles_to_buffer(
            attrs_list,
            xmin_incoming_indices, xmax_incoming_indices, ymin_incoming_indices, ymax_incoming_indices,
            xminymin_incoming_indices, xmaxymin_incoming_indices, xminymax_incoming_indices, xmaxymax_incoming_indices,
            xmin_index, xmax_index, ymin_index, ymax_index,
            xminymin_index, xmaxymin_index, xminymax_index, xmaxymax_index,
            buffer,
        )


        ibuff = 0
        attrs = typed.List([attrs_list[iattr][ipatches] for iattr in range(nattrs)])
        for ipart in range(len(pruned)):
            if ibuff >= npart_new:
                break
            if pruned[ipart]:
                for iattr in range(nattrs):
                    attrs[iattr][ipart] = buffer[ibuff, iattr]
                pruned[ipart] = False
                ibuff += 1


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
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
