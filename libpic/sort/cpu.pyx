# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
from scipy import constants
from cython cimport bool
from cython.parallel import prange
from libc.math cimport floor
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from ..utils.clist cimport CListDouble, CListBool, CListIntp

ctypedef cnp.npy_intp intp

cdef void calculate_cell_index(
    double* x, double* y, cnp.npy_bool* is_dead, 
    intp npart, intp nx, intp ny, double dx, double dy, double x0, double y0, 
    intp* particle_cell_indices, intp* grid_cell_count
) noexcept nogil:
    cdef intp ix, iy, ip, icell
    icell = 0
    for ip in range(npart):
        if not is_dead[ip]:
            ix = <intp> floor((x[ip] - x0) / dx)
            iy = <intp> floor((y[ip] - y0) / dy)
            icell = iy + ix*ny
            if 0 <= ix < nx and 0 <= iy < ny:
                particle_cell_indices[ip] = icell
                grid_cell_count[icell] += 1

        else:
            particle_cell_indices[ip] = -1
            grid_cell_count[icell] += 1

# python wrapper
def _calculate_cell_index(
    x, y, is_dead, 
    npart, nx, ny, dx, dy, x0, y0, 
    particle_cell_indices, grid_cell_count
) :

    calculate_cell_index(
        <double*>cnp.PyArray_DATA(x), <double*>cnp.PyArray_DATA(y), <cnp.npy_bool*>cnp.PyArray_DATA(is_dead),
        npart,
        nx, ny, dx, dy,
        x0, y0,
        <intp*> cnp.PyArray_DATA(particle_cell_indices), <intp*> cnp.PyArray_DATA(grid_cell_count)
    )

cdef intp cycle_sort(
    intp* cell_bound_min, intp* cell_bound_max, 
    intp nx, intp ny, 
    intp* particle_cell_indices, cnp.npy_bool* is_dead, intp* sort_idx
) noexcept nogil:
    """
    Cycle sort the particles in-place.

    `cell_bound` should be calculated first
    """
    cdef int ops = 0
    cdef int ix, iy
    cdef int ip, ip_src, ip_dst, icell_src, icell_dst, idx_dst
    
    for ix in range(nx):
        for iy in range(ny):
            icell_src = iy + ix * ny
            for ip in range(cell_bound_min[icell_src], cell_bound_max[icell_src]):
                if is_dead[ip]:
                    continue
                if particle_cell_indices[ip] == icell_src:
                    continue
                ip_src = ip
                icell_dst = particle_cell_indices[ip_src]
                idx_dst = sort_idx[ip_src]

                # when dest cell is source cell, loop is linked
                while icell_dst != icell_src:
                    # print(icell_dst, ix_dst, iy_dst, cell_bound_min[ix_dst, iy_dst], cell_bound_max[ix_dst, iy_dst])
                    for ip_dst in range(cell_bound_min[icell_dst], cell_bound_max[icell_dst]):
                        if particle_cell_indices[ip_dst] != icell_dst or is_dead[ip_dst]:
                            # print(f"{particle_cell_indices[ip_src]} at {ip_src} from [{icell}] -> {particle_cell_indices[ip_dst]} at {ip_dst} from [{icell_dst}], icell_dst={icell_dst}")
                            
                            icell_dst, particle_cell_indices[ip_dst] = particle_cell_indices[ip_dst], icell_dst
                            idx_dst, sort_idx[ip_dst] = sort_idx[ip_dst], idx_dst
                            # disp(particle_cell_indices, cell_bound_min, cell_bound_max)
                            ip_src = ip_dst

                            ops += 1
                            break
                    if is_dead[ip_dst]:
                        break
                # put back to the start of loop
                particle_cell_indices[ip] = icell_dst
                sort_idx[ip] = idx_dst
                if is_dead[ip_dst]:
                    is_dead[ip] = True
                    is_dead[ip_dst] = False

    return ops

# python wrapper
def _cycle_sort(cell_bound_min, cell_bound_max, nx, ny, particle_cell_indices, is_dead, sort_idx):
    return cycle_sort(
        <intp*> cnp.PyArray_DATA(cell_bound_min), 
        <intp*> cnp.PyArray_DATA(cell_bound_max), 
        nx, ny, 
        <intp*> cnp.PyArray_DATA(particle_cell_indices), 
        <cnp.npy_bool*> cnp.PyArray_DATA(is_dead), 
        <intp*> cnp.PyArray_DATA(sort_idx)
    )


cdef void sorted_cell_bound(
    intp* grid_cell_count, intp* cell_bound_min, intp* cell_bound_max, 
    intp nx, intp ny
)  noexcept nogil:
    cdef int icell, icell_prev
    cell_bound_min[0] = 0

    for icell in range(1, nx*ny):
        # C order       
        icell_prev = icell-1

        cell_bound_min[icell] = cell_bound_min[icell_prev] + grid_cell_count[icell_prev]
        cell_bound_max[icell_prev] = cell_bound_min[icell]
    cell_bound_max[nx*ny-1] = cell_bound_min[nx*ny-1] + grid_cell_count[nx*ny-1]
    

# python wrapper
def _sorted_cell_bound(grid_cell_count, cell_bound_min, cell_bound_max, nx, ny):
    sorted_cell_bound(
        <intp*> cnp.PyArray_DATA(grid_cell_count), 
        <intp*> cnp.PyArray_DATA(cell_bound_min), 
        <intp*> cnp.PyArray_DATA(cell_bound_max), 
        nx, ny
    )


def sort_particles_patches(
    CListIntp grid_cell_count_list, CListIntp cell_bound_min_list, CListIntp cell_bound_max_list, 
    cnp.ndarray[double, ndim=1] x0s, cnp.ndarray[double, ndim=1] y0s,
    intp nx, intp ny, double dx, double dy, 
    intp npatches,
    CListIntp particle_cell_indices_list, CListIntp sorted_indices_list, CListDouble x_list, CListDouble y_list, CListBool is_dead_list,
    CListDouble attrs_list
):
    cdef intp ipatch, icell, ip, npart, iattr
    cdef intp* grid_cell_count
    cdef intp* cell_bound_min 
    cdef intp* cell_bound_max 
    cdef double x0, y0

    cdef intp* particle_cell_indices 
    cdef intp* sorted_indices 
    cdef double* x 
    cdef double* y 
    cdef cnp.npy_bool* is_dead

    cdef double* buf
    cdef double* attr

    cdef intp nattrs = len(attrs_list) // npatches
    
    for ipatch in prange(npatches, nogil=True, schedule='runtime'):
        grid_cell_count = grid_cell_count_list.get_ptr(ipatch)
        cell_bound_min = cell_bound_min_list.get_ptr(ipatch)
        cell_bound_max = cell_bound_max_list.get_ptr(ipatch)
        x0 = x0s[ipatch]
        y0 = y0s[ipatch]
        particle_cell_indices = particle_cell_indices_list.get_ptr(ipatch)
        sorted_indices = sorted_indices_list.get_ptr(ipatch)
        x = x_list.get_ptr(ipatch)
        y = y_list.get_ptr(ipatch)
        is_dead = is_dead_list.get_ptr(ipatch)
        
        npart = is_dead_list.get_size(ipatch)
        
        # set 0
        for icell in range(nx*ny):
            grid_cell_count[icell] = 0

        calculate_cell_index(
            x, y, is_dead,
            npart,
            nx, ny, dx, dy,
            x0, y0,
            particle_cell_indices, grid_cell_count
        )

        sorted_cell_bound(
            grid_cell_count, 
            cell_bound_min, 
            cell_bound_max, 
            nx, ny
        )
        

        for ip in range(npart):
            sorted_indices[ip] = ip
        cycle_sort(cell_bound_min, cell_bound_max, nx, ny, particle_cell_indices, is_dead, sorted_indices)
        
        buf = <double*> malloc(npart * sizeof(double))
        for iattr in range(nattrs):
            attr = attrs_list.get_ptr(iattr+ipatch*nattrs)
            for ip in range(npart):
                if ip != sorted_indices[ip]:
                    buf[ip] = attr[sorted_indices[ip]]
            for ip in range(npart):
                if ip != sorted_indices[ip]:
                    attr[ip] = buf[ip]
        

        free(buf)