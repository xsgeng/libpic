# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
from scipy import constants
from cython cimport bool
from cython.parallel import prange
from libc.math cimport floor
import numpy as np
cimport numpy as cnp

from ..utils.clist cimport CListDouble, CListBool

cdef double c = constants.c
cdef double one_third = 1./3.

cdef void calculate_S(double delta, int shift, double* S) noexcept nogil:
    delta2 = delta * delta

    delta_minus    = 0.5 * ( delta2+delta+0.25 )
    delta_mid      = 0.75 - delta2
    delta_positive = 0.5 * ( delta2-delta+0.25 )

    cdef bint minus = shift == -1
    cdef bint mid = shift == 0
    cdef bint positive = shift == 1

    S[0] = minus * delta_minus
    S[1] = minus * delta_mid      + mid * delta_minus
    S[2] = minus * delta_positive + mid * delta_mid      + positive * delta_minus
    S[3] =                          mid * delta_positive + positive * delta_mid
    S[4] =                                                 positive * delta_positive
    

# NOTE ~True = -1 for npy_bool
cdef void current_deposit_2d(
    double* rho, double* jx, double* jy, double* jz, 
    double* x, double* y, double* ux, double* uy, double* uz, double* inv_gamma, 
    cnp.npy_bool* is_dead, 
    cnp.npy_intp npart, cnp.npy_intp nx, cnp.npy_intp ny,
    double dx, double dy, double x0, double y0, double dt, double* w, double q
) noexcept nogil:
    cdef double x_old, y_old, x_adv, y_adv
    cdef double[5] S0x, S1x, S0y, S1y, DSx, DSy, jy_buff
    
    cdef cnp.npy_intp i, j, ipart
    
    cdef int dcell_x, dcell_y, ix0, iy0, ix1, iy1, ix, iy
    
    cdef double vx, vy, vz
    
    cdef double x_over_dx0, x_over_dx1, y_over_dy0, y_over_dy1
    
    cdef double charge_density, factor, jx_buff
    for ipart in range(npart):
        if is_dead[ipart]:
            continue
        vx = ux[ipart]*c*inv_gamma[ipart]
        vy = uy[ipart]*c*inv_gamma[ipart]
        vz = uz[ipart]*c*inv_gamma[ipart] if not is_dead[ipart] else 0.0
        x_old = x[ipart] - vx*0.5*dt - x0
        y_old = y[ipart] - vy*0.5*dt - y0
        x_adv = x[ipart] + vx*0.5*dt - x0
        y_adv = y[ipart] + vy*0.5*dt - y0

        # positions at t + dt/2, before pusher
        # +0.5 for cell-centered coordinate
        x_over_dx0 = x_old / dx
        ix0 = <int>floor(x_over_dx0+0.5)
        y_over_dy0 = y_old / dy
        iy0 = <int>floor(y_over_dy0+0.5)

        calculate_S(ix0 - x_over_dx0, 0, S0x) #gx
        calculate_S(iy0 - y_over_dy0, 0, S0y)

        # positions at t + 3/2*dt, after pusher
        x_over_dx1 = x_adv / dx
        ix1 = <int>floor(x_over_dx1+0.5)
        dcell_x = ix1 - ix0

        y_over_dy1 = y_adv / dy
        iy1 = <int>floor(y_over_dy1+0.5)
        dcell_y = iy1 - iy0
        

        calculate_S(ix1 - x_over_dx1, dcell_x, S1x)
        calculate_S(iy1 - y_over_dy1, dcell_y, S1y)

        for i in range(5):
            DSx[i] = S1x[i] - S0x[i]
            DSy[i] = S1y[i] - S0y[i]
            jy_buff[i] = 0

        charge_density = q * w[ipart] / (dx*dy) if not is_dead[ipart] else 0.0
        factor = charge_density / dt


        # i and j are the relative shift, 0-based index
        # [0,   1, 2, 3, 4]
        #     [-1, 0, 1, 2] for dcell = 1;
        #     [-1, 0, 1] for dcell_ = 0
        # [-2, -1, 0, 1] for dcell = -1
        for j in range(min(1, 1+dcell_y), max(4, 4+dcell_y)):
            jx_buff = 0.0
            iy = iy0 + (j - 2)
            if iy < 0:
                iy = ny + iy
            for i in range(min(1, 1+dcell_x), max(4, 4+dcell_x)):
                ix = ix0 + (i - 2)
                if ix < 0:
                    ix = nx + ix
                wx = DSx[i] * (S0y[j] + 0.5 * DSy[j])
                wy = DSy[j] * (S0x[i] + 0.5 * DSx[i])
                wz = S0x[i] * S0y[j] + 0.5 * DSx[i] * S0y[j] \
                    + 0.5 * S0x[i] * DSy[j] + one_third * DSx[i] * DSy[j]

                jx_buff -= factor * dx * wx
                jy_buff[i] -= factor * dy * wy

                jx[iy + ny*ix] += jx_buff
                jy[iy + ny*ix] += jy_buff[i]
                jz[iy + ny*ix] += factor*dt * wz * vz
                rho[iy + ny*ix] += charge_density * S1x[i] * S1y[j]

def current_deposition_cpu(
    CListDouble rho_list,
    CListDouble jx_list, CListDouble jy_list, CListDouble jz_list,
    double[:] x0_list, double[:] y0_list,
    CListDouble x_list, CListDouble y_list, CListDouble ux_list, CListDouble uy_list, CListDouble uz_list,
    CListDouble inv_gamma_list,
    CListBool is_dead_list,
    cnp.npy_intp npatches,
    double dx, double dy, double dt, CListDouble w_list, double q,
):
    cdef cnp.npy_intp ipatch

    cdef double* rho
    cdef double* jx
    cdef double* jy
    cdef double* jz
    cdef double* x
    cdef double* y
    cdef double* ux
    cdef double* uy
    cdef double* uz
    cdef double* inv_gamma
    cdef cnp.npy_bool* is_dead

    cdef double x0, y0
    cdef cnp.npy_intp npart, nx, ny
    for ipatch in prange(npatches, nogil=True, schedule='dynamic', chunksize=1):
        rho = rho_list.get_ptr(ipatch)
        jx = jx_list.get_ptr(ipatch)
        jy = jy_list.get_ptr(ipatch)
        jz = jz_list.get_ptr(ipatch)
        x = x_list.get_ptr(ipatch)
        y = y_list.get_ptr(ipatch)
        ux = ux_list.get_ptr(ipatch)
        uy = uy_list.get_ptr(ipatch)
        uz = uz_list.get_ptr(ipatch)
        w = w_list.get_ptr(ipatch)
        inv_gamma = inv_gamma_list.get_ptr(ipatch)
        is_dead = is_dead_list.get_ptr(ipatch)
        npart = is_dead_list.get_size(ipatch)

        x0 = x0_list[ipatch]
        y0 = y0_list[ipatch]

        nx = jx_list.get_shape(ipatch)[0]
        ny = jx_list.get_shape(ipatch)[1]
        current_deposit_2d(
            rho, jx, jy, jz, 
            x, y, ux, uy, uz, inv_gamma, 
            is_dead, 
            npart, nx, ny,
            dx, dy, x0, y0, dt, w, q
        )
