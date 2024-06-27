import numpy as np
from numba import jit, njit, prange, typed

@njit
def calculate_cell_index(x, y, pruned, nx, ny, dx, dy, x0, y0, particle_cell_indices, grid_cell_count):
    ix = 0
    iy = 0
    for ip in range(len(x)):
        if not pruned[ip]:
            ix = int(np.floor((x[ip] - x0) / dx))
            iy = int(np.floor((y[ip] - y0) / dy))
            if 0 <= ix < nx and 0 <= iy < ny:
                particle_cell_indices[ip] = iy + ix * ny
                grid_cell_count[ix, iy] += 1

        else:
            particle_cell_indices[ip] = -1
            grid_cell_count[ix, iy] += 1
            
@njit
def cycle_sort(cell_bound_min, cell_bound_max, nx, ny, particle_cell_indices, pruned, sort_idx):
    """
    Cycle sort the particles in-place.

    `cell_bound` should be calculated first
    """
    # nattr = len(attrs)
    # buf, will be stored to the dest particle
    # attr_dst = np.zeros(nattr)
    for ix in range(nx):
        for iy in range(ny):
            icell_src = iy + ix * ny
            for ip in range(cell_bound_min[ix, iy], cell_bound_max[ix, iy]):
                if pruned[ip]:
                    continue
                if particle_cell_indices[ip] == icell_src:
                    continue
                ip_src = ip
                icell_dst = particle_cell_indices[ip_src]
                idx_dst = sort_idx[ip_src]

                # when dest cell is source cell, loop is linked
                while icell_dst != icell_src:
                    ix_dst = icell_dst // ny
                    iy_dst = icell_dst % ny
                    # print(icell_dst, ix_dst, iy_dst, cell_bound_min[ix_dst, iy_dst], cell_bound_max[ix_dst, iy_dst])
                    for ip_dst in range(cell_bound_min[ix_dst, iy_dst], cell_bound_max[ix_dst, iy_dst]):
                        if particle_cell_indices[ip_dst] != icell_dst or pruned[ip_dst]:
                            # print(f"{particle_cell_indices[ip_src]} at {ip_src} from [{icell}] -> {particle_cell_indices[ip_dst]} at {ip_dst} from [{icell_dst}], icell_dst={icell_dst}")
                            
                            icell_dst, particle_cell_indices[ip_dst] = particle_cell_indices[ip_dst], icell_dst
                            idx_dst, sort_idx[ip_dst] = sort_idx[ip_dst], idx_dst
                            # disp(particle_cell_indices, cell_bound_min, cell_bound_max)
                            ip_src = ip_dst
                            break
                    if pruned[ip_dst]:
                        break
                # put back to the start of loop
                particle_cell_indices[ip] = icell_dst
                sort_idx[ip] = idx_dst
                if pruned[ip_dst]:
                    pruned[ip] = True
                    pruned[ip_dst] = False

@njit
def sorted_cell_bound(grid_cell_count, cell_bound_min, cell_bound_max, nx, ny):
    cell_bound_min[0, 0] = 0

    for icell in range(1, nx*ny):
        # C order
        iy = icell % ny
        ix = icell // ny

        iy_prev = (icell-1) % ny
        ix_prev = (icell-1) // ny

        cell_bound_min[ix, iy] = cell_bound_min[ix_prev, iy_prev] + grid_cell_count[ix_prev, iy_prev]
        cell_bound_max[ix_prev, iy_prev] = cell_bound_min[ix, iy]
    cell_bound_max[nx-1, ny-1] = cell_bound_min[nx-1, ny-1] + grid_cell_count[nx-1, ny-1]

@njit
def cycle_sort_attrs(attrs, sorted_indices):
    n = len(sorted_indices)
    for i in range(n):
        if sorted_indices[i] != i:
            temp = attrs[i]
            j = i
            while sorted_indices[j] != i:
                k = sorted_indices[j]
                attrs[j] = attrs[k]
                sorted_indices[j] = j
                j = k
            attrs[j] = temp
            sorted_indices[j] = j
            
@njit(parallel=True, cache=True)
def sort_particles_patches(
    grid_cell_count_list, cell_bound_min_list, cell_bound_max_list, x0s, y0s,
    nx, ny, dx, dy, particle_cell_indices_list, sorted_indices_list, x_list, y_list, pruned_list, *attrs_list
):
    nattrs = len(attrs_list)
    
    for ipatch in prange(len(grid_cell_count_list)):
        grid_cell_count_list[ipatch].fill(0)
        calculate_cell_index(
            x_list[ipatch], y_list[ipatch], pruned_list[ipatch],
            nx, ny, dx, dy,
            x0s[ipatch], y0s[ipatch],
            particle_cell_indices_list[ipatch], grid_cell_count_list[ipatch]
        )
        sorted_cell_bound(
            grid_cell_count_list[ipatch], 
            cell_bound_min_list[ipatch], 
            cell_bound_max_list[ipatch], 
            nx, ny
        )

        sorted_indices = sorted_indices_list[ipatch]
        npart = x_list[ipatch].size
        for ip in range(npart):
            sorted_indices[ip] = ip
        cycle_sort(cell_bound_min_list[ipatch], cell_bound_max_list[ipatch], nx, ny, particle_cell_indices_list[ipatch], pruned_list[ipatch], sorted_indices)
        attrs = typed.List([attrs_list[iattr][ipatch] for iattr in range(nattrs)])
        nattrs = len(attrs)
        for iattr in range(nattrs):
            attrs[iattr][:] = attrs[iattr][sorted_indices]
