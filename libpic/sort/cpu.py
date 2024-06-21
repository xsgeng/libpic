import numpy as np
from numba import njit, prange, typed

@njit
def calculate_cell_index(x, y, nx, ny, dx, dy, x0, y0, particle_cell_indices, grid_cell_count):
    for ip in range(len(x)):
        ix = int(np.floor((x[ip] - x0) / dx))
        iy = int(np.floor((y[ip] - y0) / dy))
        if 0 <= ix < nx and 0 <= iy < ny:
            particle_cell_indices[ip] = iy + ix * ny
            grid_cell_count[ix, iy] += 1

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

        
@njit(parallel=True, cache=True)
def sort_particles_patches(
    grid_cell_count_list, cell_bound_min_list, cell_bound_max_list, x0s, y0s,
    nx, ny, dx, dy, particle_cell_indices_list, x_list, y_list, *attrs_list
):
    nattrs = len(attrs_list)
    
    for ipatch in prange(len(grid_cell_count_list)):
        grid_cell_count_list[ipatch].fill(0)
        calculate_cell_index(
            x_list[ipatch], y_list[ipatch], nx, ny, dx, dy,
            x0s[ipatch], y0s[ipatch],
            particle_cell_indices_list[ipatch], grid_cell_count_list[ipatch]
        )
        
        attrs = typed.List([attrs_list[iattr][ipatch] for iattr in range(nattrs)])
        sort_idx = np.argsort(particle_cell_indices_list[ipatch])
        x_list[ipatch][:] = x_list[ipatch][sort_idx]
        y_list[ipatch][:] = y_list[ipatch][sort_idx]
        nattrs = len(attrs)
        for iattr in range(nattrs):
            attr = attrs[iattr]
            attr[:] = attr[sort_idx]
        
        calculate_cell_index(
            x_list[ipatch], y_list[ipatch], nx, ny, dx, dy,
            x0s[ipatch], y0s[ipatch],
            particle_cell_indices_list[ipatch], grid_cell_count_list[ipatch]
        )
        