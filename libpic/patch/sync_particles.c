#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include "../utils/cutils.h"

// #ifdef USE_TCMALLOC
// #include "tcmalloc.h"
// #define malloc tc_malloc
// #define free tc_free
// #endif

// Type-independent cleanup function using void*
static void cleanup_ptr(void* p) {
    void** ptr = (void**)p;
    if (*ptr) free(*ptr);
    *ptr = NULL;
}
#define AUTOFREE __attribute__((cleanup(cleanup_ptr)))

// Implementation of count_outgoing_particles function
static void count_outgoing_particles(
    double* x, double* y, 
    double xmin, double xmax, double ymin, double ymax,
    npy_intp npart,
    npy_intp* npart_xmin, npy_intp* npart_xmax, 
    npy_intp* npart_ymin, npy_intp* npart_ymax,
    npy_intp* npart_xminymin, npy_intp* npart_xmaxymin, 
    npy_intp* npart_xminymax, npy_intp* npart_xmaxymax
) {
    *npart_xmin = 0;
    *npart_xmax = 0;
    *npart_ymin = 0;
    *npart_ymax = 0;
    *npart_xminymin = 0;
    *npart_xmaxymin = 0;
    *npart_xminymax = 0;
    *npart_xmaxymax = 0;

    for (npy_intp ip = 0; ip < npart; ip++) {
        if (y[ip] < ymin) {
            if (x[ip] < xmin) {
                (*npart_xminymin)++;
                continue;
            }
            else if (x[ip] > xmax) {
                (*npart_xmaxymin)++;
                continue;
            }
            else {
                (*npart_ymin)++;
                continue;
            }
        }
        else if (y[ip] > ymax) {
            if (x[ip] < xmin) {
                (*npart_xminymax)++;
                continue;
            }
            else if (x[ip] > xmax) {
                (*npart_xmaxymax)++;
                continue;
            }
            else {
                (*npart_ymax)++;
                continue;
            }
        }
        else {
            if (x[ip] < xmin) {
                (*npart_xmin)++;
                continue;
            }
            else if (x[ip] > xmax) {
                (*npart_xmax)++;
                continue;
            }
        }
    }
}

// Get indices of incoming particles from neighboring patches
static void get_incoming_index(
    double** x_list, double** y_list, npy_intp* npart_list,
    double* xmin_list, double* xmax_list, double* ymin_list, double* ymax_list,
    npy_intp xmin_index, npy_intp xmax_index,
    npy_intp ymin_index, npy_intp ymax_index,
    npy_intp xminymin_index, npy_intp xmaxymin_index,
    npy_intp xminymax_index, npy_intp xmaxymax_index,
    // out
    npy_intp* xmin_indices, npy_intp* xmax_indices,
    npy_intp* ymin_indices, npy_intp* ymax_indices,
    npy_intp* xminymin_indices, npy_intp* xmaxymin_indices,
    npy_intp* xminymax_indices, npy_intp* xmaxymax_indices
) {
    // On xmin boundary
    if (xmin_index >= 0) {
        double* x_on_xmin = x_list[xmin_index];
        double* y_on_xmin = y_list[xmin_index];
        
        double xmax = xmax_list[xmin_index];
        double ymin = ymin_list[xmin_index];
        double ymax = ymax_list[xmin_index];

        npy_intp npart = npart_list[xmin_index];
        
        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmin[ipart] > xmax) && (y_on_xmin[ipart] >= ymin) && (y_on_xmin[ipart] <= ymax)) {
                xmin_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xmax boundary
    if (xmax_index >= 0) {
        double* x_on_xmax = x_list[xmax_index];
        double* y_on_xmax = y_list[xmax_index];
        
        double xmin = xmin_list[xmax_index];
        double ymin = ymin_list[xmax_index];
        double ymax = ymax_list[xmax_index];

        npy_intp npart = npart_list[xmax_index];
        
        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmax[ipart] < xmin) && (y_on_xmax[ipart] >= ymin) && (y_on_xmax[ipart] <= ymax)) {
                xmax_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On ymin boundary
    if (ymin_index >= 0) {
        double* x_on_ymin = x_list[ymin_index];
        double* y_on_ymin = y_list[ymin_index];
        
        double xmin = xmin_list[ymin_index];
        double xmax = xmax_list[ymin_index];
        double ymax = ymax_list[ymin_index];

        npy_intp npart = npart_list[ymin_index];
        
        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_ymin[ipart] >= xmin) && (x_on_ymin[ipart] <= xmax) && (y_on_ymin[ipart] > ymax)) {
                ymin_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On ymax boundary
    if (ymax_index >= 0) {
        double* x_on_ymax = x_list[ymax_index];
        double* y_on_ymax = y_list[ymax_index];
        
        double xmin = xmin_list[ymax_index];
        double xmax = xmax_list[ymax_index];
        double ymin = ymin_list[ymax_index];

        npy_intp npart = npart_list[ymax_index];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_ymax[ipart] >= xmin) && (x_on_ymax[ipart] <= xmax) && (y_on_ymax[ipart] < ymin)) {
                ymax_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xminymin boundary
    if (xminymin_index >= 0) {
        double* x_on_xminymin = x_list[xminymin_index];
        double* y_on_xminymin = y_list[xminymin_index];
        
        double xmax = xmax_list[xminymin_index];
        double ymax = ymax_list[xminymin_index];

        npy_intp npart = npart_list[xminymin_index];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xminymin[ipart] > xmax) && (y_on_xminymin[ipart] > ymax)) {
                xminymin_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xmaxymin boundary
    if (xmaxymin_index >= 0) {
        double* x_on_xmaxymin = x_list[xmaxymin_index];
        double* y_on_xmaxymin = y_list[xmaxymin_index];
        
        double xmin = xmin_list[xmaxymin_index];
        double ymax = ymax_list[xmaxymin_index];

        npy_intp npart = npart_list[xmaxymin_index];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmaxymin[ipart] < xmin) && (y_on_xmaxymin[ipart] > ymax)) {
                xmaxymin_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xminymax boundary
    if (xminymax_index >= 0) {
        double* x_on_xminymax = x_list[xminymax_index];
        double* y_on_xminymax = y_list[xminymax_index];
        
        double xmax = xmax_list[xminymax_index];
        double ymin = ymin_list[xminymax_index];

        npy_intp npart = npart_list[xminymax_index];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xminymax[ipart] > xmax) && (y_on_xminymax[ipart] < ymin)) {
                xminymax_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xmaxymax boundary
    if (xmaxymax_index >= 0) {
        double* x_on_xmaxymax = x_list[xmaxymax_index];
        double* y_on_xmaxymax = y_list[xmaxymax_index];
        
        double xmin = xmin_list[xmaxymax_index];
        double ymin = ymin_list[xmaxymax_index];

        npy_intp npart = npart_list[xmaxymax_index];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmaxymax[ipart] < xmin) && (y_on_xmaxymax[ipart] < ymin)) {
                xmaxymax_indices[i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
}

// Fill buffer with boundary particles
static void fill_boundary_particles_to_buffer(
    double** attrs_list, npy_intp nattrs,
    npy_intp* xmin_indices, npy_intp* xmax_indices,
    npy_intp* ymin_indices, npy_intp* ymax_indices,
    npy_intp* xminymin_indices, npy_intp* xmaxymin_indices,
    npy_intp* xminymax_indices, npy_intp* xmaxymax_indices,
    npy_intp npart_xmin, npy_intp npart_xmax,
    npy_intp npart_ymin, npy_intp npart_ymax,
    npy_intp npart_xminymin, npy_intp npart_xmaxymin,
    npy_intp npart_xminymax, npy_intp npart_xmaxymax,
    npy_intp xmin_index, npy_intp xmax_index,
    npy_intp ymin_index, npy_intp ymax_index,
    npy_intp xminymin_index, npy_intp xmaxymin_index,
    npy_intp xminymax_index, npy_intp xmaxymax_index,
    double* buffer
) {
    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
        npy_intp ibuff = 0;
        
        // From xmin boundary
        if (xmin_index >= 0) {
            double* attr = attrs_list[xmin_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_xmin; i++) {
                npy_intp idx = xmin_indices[i];
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From xmax boundary
        if (xmax_index >= 0) {
            double* attr = attrs_list[xmax_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_xmax; i++) {
                npy_intp idx = xmax_indices[i];
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From ymin boundary
        if (ymin_index >= 0) {
            double* attr = attrs_list[ymin_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_ymin; i++) {
                npy_intp idx = ymin_indices[i];
                if (idx == 0 && i > 0) break; // Assuming 0 is a sentinel value
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From ymax boundary
        if (ymax_index >= 0) {
            double* attr = attrs_list[ymax_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_ymax; i++) {
                npy_intp idx = ymax_indices[i];
                if (idx == 0 && i > 0) break; // Assuming 0 is a sentinel value
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From corners
        // From xminymin boundary
        if (xminymin_index >= 0) {
            double* attr = attrs_list[xminymin_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_xminymin; i++) {
                npy_intp idx = xminymin_indices[i];
                if (idx == 0 && i > 0) break; // Assuming 0 is a sentinel value
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From xmaxymin boundary
        if (xmaxymin_index >= 0) {
            double* attr = attrs_list[xmaxymin_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_xmaxymin; i++) {
                npy_intp idx = xmaxymin_indices[i];
                if (idx == 0 && i > 0) break; // Assuming 0 is a sentinel value
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From xminymax boundary
        if (xminymax_index >= 0) {
            double* attr = attrs_list[xminymax_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_xminymax; i++) {
                npy_intp idx = xminymax_indices[i];
                if (idx == 0 && i > 0) break; // Assuming 0 is a sentinel value
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
        
        // From xmaxymax boundary
        if (xmaxymax_index >= 0) {
            double* attr = attrs_list[xmaxymax_index*nattrs + iattr];
            for (npy_intp i = 0; i < npart_xmaxymax; i++) {
                npy_intp idx = xmaxymax_indices[i];
                if (idx == 0 && i > 0) break; // Assuming 0 is a sentinel value
                buffer[ibuff*nattrs+iattr] = attr[idx];
                ibuff++;
            }
        }
    }
}



// Mark out-of-bound particles as dead
static void mark_out_of_bound_as_dead(
    double *x, double *y, npy_bool *is_dead, npy_intp npart, 
    double xmin, double xmax, 
    double ymin, double ymax,
    npy_intp npatches
) {
    for (npy_intp ipart = 0; ipart < npart; ipart++) {
        if (is_dead[ipart]) {
            x[ipart] = NAN;
            y[ipart] = NAN;
            continue;
        }
        if (x[ipart] < xmin || x[ipart] > xmax || y[ipart] < ymin || y[ipart] > ymax) {
            is_dead[ipart] = 1;
            x[ipart] = NAN;
            y[ipart] = NAN;
        }
    }
}

PyObject* get_npart_to_extend(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject* particles_list;
    PyObject* patch_list;
    double dx, dy;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOndd", 
            &particles_list, 
            &patch_list,
            &npatches,
            &dx, &dy
        )
    ) {
        return NULL;
    }

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");

    AUTOFREE npy_intp *xmin_index_list = get_attr_int(patch_list, npatches, "xmin_neighbor_index");
    AUTOFREE npy_intp *xmax_index_list = get_attr_int(patch_list, npatches, "xmax_neighbor_index");
    AUTOFREE npy_intp *ymin_index_list = get_attr_int(patch_list, npatches, "ymin_neighbor_index");
    AUTOFREE npy_intp *ymax_index_list = get_attr_int(patch_list, npatches, "ymax_neighbor_index");
    AUTOFREE npy_intp *xminymin_index_list = get_attr_int(patch_list, npatches, "xminymin_neighbor_index");
    AUTOFREE npy_intp *xmaxymin_index_list = get_attr_int(patch_list, npatches, "xmaxymin_neighbor_index");
    AUTOFREE npy_intp *xminymax_index_list = get_attr_int(patch_list, npatches, "xminymax_neighbor_index");
    AUTOFREE npy_intp *xmaxymax_index_list = get_attr_int(patch_list, npatches, "xmaxymax_neighbor_index");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");

    // Adjust particle boundaries
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
    }

    // Allocate arrays for particle counts with cleanup attributes
    npy_intp dims = 8*npatches;
    PyArrayObject *npart_to_extend_array = (PyArrayObject*) PyArray_Zeros(1, &npatches, NPY_INT64, NPY_CARRAY);
    PyArrayObject *npart_incoming_array = (PyArrayObject*) PyArray_Zeros(1, &npatches, NPY_INT64, NPY_CARRAY);
    PyArrayObject *npart_outgoing_array = (PyArrayObject*) PyArray_Zeros(1, &dims, NPY_INT64, NPY_CARRAY);

    npy_intp *npart_to_extend = (npy_intp*) PyArray_DATA(npart_to_extend_array);
    npy_intp *npart_incoming = (npy_intp*) PyArray_DATA(npart_incoming_array);
    npy_intp *npart_outgoing = (npy_intp*) PyArray_DATA(npart_outgoing_array);

    // Count outgoing particles for each patch
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        double *x = x_list[ipatch];
        double *y = y_list[ipatch];
        double xmin = xmin_list[ipatch];
        double xmax = xmax_list[ipatch];
        double ymin = ymin_list[ipatch];
        double ymax = ymax_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        
        // Count particles going out of bounds
        npy_intp npart_xmin, npart_xmax, npart_ymin, npart_ymax;
        npy_intp npart_xminymin, npart_xmaxymin, npart_xminymax, npart_xmaxymax;
        
        count_outgoing_particles(
            x, y, xmin, xmax, ymin, ymax, npart,
            &npart_xmin, &npart_xmax, &npart_ymin, &npart_ymax,
            &npart_xminymin, &npart_xmaxymin, &npart_xminymax, &npart_xmaxymax
        );
        
        // Store results in the outgoing array
        
        npart_outgoing[ipatch * 8 + 0] = npart_xmin;
        npart_outgoing[ipatch * 8 + 1] = npart_xmax;
        npart_outgoing[ipatch * 8 + 2] = npart_ymin;
        npart_outgoing[ipatch * 8 + 3] = npart_ymax;
        npart_outgoing[ipatch * 8 + 4] = npart_xminymin;
        npart_outgoing[ipatch * 8 + 5] = npart_xmaxymin;
        npart_outgoing[ipatch * 8 + 6] = npart_xminymax;
        npart_outgoing[ipatch * 8 + 7] = npart_xmaxymax;
    }
    
    // Calculate incoming particles for each patch
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_bool *is_dead = is_dead_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        
        npy_intp xmin_index = xmin_index_list[ipatch];
        npy_intp xmax_index = xmax_index_list[ipatch];
        npy_intp ymin_index = ymin_index_list[ipatch];
        npy_intp ymax_index = ymax_index_list[ipatch];
        
        // Calculate corner indices
        npy_intp xminymin_index = xminymin_index_list[ipatch];
        npy_intp xmaxymin_index = xmaxymin_index_list[ipatch];
        npy_intp xminymax_index = xminymax_index_list[ipatch];
        npy_intp xmaxymax_index = xmaxymax_index_list[ipatch];
        
        // Count incoming particles
        npy_intp npart_new = 0;
        
        if (xmax_index >= 0) {
            npart_new += npart_outgoing[0 * npatches + xmax_index];
        }
        if (xmin_index >= 0) {
            npart_new += npart_outgoing[1 * npatches + xmin_index];
        }
        if (ymax_index >= 0) {
            npart_new += npart_outgoing[2 * npatches + ymax_index];
        }
        if (ymin_index >= 0) {
            npart_new += npart_outgoing[3 * npatches + ymin_index];
        }
        
        // Corners
        if (xmaxymax_index >= 0) {
            npart_new += npart_outgoing[4 * npatches + xmaxymax_index];
        }
        if (xminymax_index >= 0) {
            npart_new += npart_outgoing[5 * npatches + xminymax_index];
        }
        if (xmaxymin_index >= 0) {
            npart_new += npart_outgoing[6 * npatches + xmaxymin_index];
        }
        if (xminymin_index >= 0) {
            npart_new += npart_outgoing[7 * npatches + xminymin_index];
        }
        
        // Count dead particles
        npy_intp ndead = 0;
        for (npy_intp i = 0; i < npart; i++) {
            if (is_dead[i]) {
                ndead++;
            }
        }
        
        // Calculate number of particles to extend
        if ((npart_new - ndead) > 0) {
            // Reserve more space for new particles (25% extra)
            npart_to_extend[ipatch] = npart_new - ndead + (npy_intp)(npart * 0.25);
        }
        
        npart_incoming[ipatch] = npart_new;
    }
    Py_END_ALLOW_THREADS





    
    PyObject *ret = PyTuple_Pack(3, npart_to_extend_array, npart_incoming_array, npart_outgoing_array);
    return ret;
}

// Fill particles from boundary
PyObject* fill_particles_from_boundary(PyObject* self, PyObject* args) {
    // Parse input arguments
    AUTOFREE PyObject *particles_list, *patch_list, *attrs;
    AUTOFREE PyArrayObject *npart_incoming_array, *npart_outgoing_array;
    double dx, dy;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOOOnddO", 
            &particles_list, 
            &patch_list,
            &npart_incoming_array,
            &npart_outgoing_array,
            &npatches,
            &dx, &dy, &attrs
        )
    ) {
        return NULL;
    }

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");

    AUTOFREE npy_intp *xmin_index_list = get_attr_int(patch_list, npatches, "xmin_neighbor_index");
    AUTOFREE npy_intp *xmax_index_list = get_attr_int(patch_list, npatches, "xmax_neighbor_index");
    AUTOFREE npy_intp *ymin_index_list = get_attr_int(patch_list, npatches, "ymin_neighbor_index");
    AUTOFREE npy_intp *ymax_index_list = get_attr_int(patch_list, npatches, "ymax_neighbor_index");
    AUTOFREE npy_intp *xminymin_index_list = get_attr_int(patch_list, npatches, "xminymin_neighbor_index");
    AUTOFREE npy_intp *xmaxymin_index_list = get_attr_int(patch_list, npatches, "xmaxymin_neighbor_index");
    AUTOFREE npy_intp *xminymax_index_list = get_attr_int(patch_list, npatches, "xminymax_neighbor_index");
    AUTOFREE npy_intp *xmaxymax_index_list = get_attr_int(patch_list, npatches, "xmaxymax_neighbor_index");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");

    AUTOFREE npy_intp *npart_incoming = PyArray_DATA(npart_incoming_array);
    AUTOFREE npy_intp *npart_outgoing = PyArray_DATA(npart_outgoing_array);

    npy_intp nattrs = PyList_Size(attrs);
    
    // Create array of attribute arrays
    AUTOFREE double **attrs_list = malloc(nattrs * npatches * sizeof(double*));
    for (Py_ssize_t iattr = 0; iattr < nattrs; iattr++) {
        PyObject *attr_name = PyList_GetItem(attrs, iattr);
        
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            PyObject *particle = PyList_GetItem(particles_list, ipatch);
            PyObject *attr_array = PyObject_GetAttr(particle, attr_name);
            attrs_list[ipatch*nattrs + iattr] = (double*) PyArray_DATA((PyArrayObject*)attr_array);
            Py_DECREF(attr_array);
        }
    }

    NPY_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_intp npart_new = npart_incoming[ipatch];
        if (npart_new <= 0) {
            continue;
        }
        
        npy_bool* is_dead = is_dead_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        
        npy_intp xmin_index = xmin_index_list[ipatch];
        npy_intp xmax_index = xmax_index_list[ipatch];
        npy_intp ymin_index = ymin_index_list[ipatch];
        npy_intp ymax_index = ymax_index_list[ipatch];
        
        // Calculate corner indices
        npy_intp xminymin_index = xminymin_index_list[ipatch];
        npy_intp xmaxymin_index = xmaxymin_index_list[ipatch];
        npy_intp xminymax_index = xminymax_index_list[ipatch];
        npy_intp xmaxymax_index = xmaxymax_index_list[ipatch];
        
        // Number of particles coming from each boundary
        npy_intp npart_incoming_xmax = (xmax_index >= 0) ? npart_outgoing[0 * npatches + xmax_index] : 0;
        npy_intp npart_incoming_xmin = (xmin_index >= 0) ? npart_outgoing[1 * npatches + xmin_index] : 0;
        npy_intp npart_incoming_ymax = (ymax_index >= 0) ? npart_outgoing[2 * npatches + ymax_index] : 0;
        npy_intp npart_incoming_ymin = (ymin_index >= 0) ? npart_outgoing[3 * npatches + ymin_index] : 0;
        
        // Corners
        npy_intp npart_incoming_xmaxymax = (xmaxymax_index >= 0) ? npart_outgoing[4 * npatches + xmaxymax_index] : 0;
        npy_intp npart_incoming_xminymax = (xminymax_index >= 0) ? npart_outgoing[5 * npatches + xminymax_index] : 0;
        npy_intp npart_incoming_xmaxymin = (xmaxymin_index >= 0) ? npart_outgoing[6 * npatches + xmaxymin_index] : 0;
        npy_intp npart_incoming_xminymin = (xminymin_index >= 0) ? npart_outgoing[7 * npatches + xminymin_index] : 0;
        
        // Indices of particles coming from boundary
        AUTOFREE npy_intp* xmin_incoming_indices = NULL;
        AUTOFREE npy_intp* xmax_incoming_indices = NULL;
        AUTOFREE npy_intp* ymin_incoming_indices = NULL;
        AUTOFREE npy_intp* ymax_incoming_indices = NULL;
        AUTOFREE npy_intp* xminymin_incoming_indices = NULL;
        AUTOFREE npy_intp* xmaxymin_incoming_indices = NULL;
        AUTOFREE npy_intp* xminymax_incoming_indices = NULL;
        AUTOFREE npy_intp* xmaxymax_incoming_indices = NULL;
        
        // Allocate memory for indices
        if (npart_incoming_xmin > 0) {
            xmin_incoming_indices = (npy_intp*)malloc(npart_incoming_xmin * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_xmin; i++) xmin_incoming_indices[i] = 0;
        }
        if (npart_incoming_xmax > 0) {
            xmax_incoming_indices = (npy_intp*)malloc(npart_incoming_xmax * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_xmax; i++) xmax_incoming_indices[i] = 0;
        }
        if (npart_incoming_ymin > 0) {
            ymin_incoming_indices = (npy_intp*)malloc(npart_incoming_ymin * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_ymin; i++) ymin_incoming_indices[i] = 0;
        }
        if (npart_incoming_ymax > 0) {
            ymax_incoming_indices = (npy_intp*)malloc(npart_incoming_ymax * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_ymax; i++) ymax_incoming_indices[i] = 0;
        }
        if (npart_incoming_xminymin > 0) {
            xminymin_incoming_indices = (npy_intp*)malloc(npart_incoming_xminymin * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_xminymin; i++) xminymin_incoming_indices[i] = 0;
        }
        if (npart_incoming_xmaxymin > 0) {
            xmaxymin_incoming_indices = (npy_intp*)malloc(npart_incoming_xmaxymin * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_xmaxymin; i++) xmaxymin_incoming_indices[i] = 0;
        }
        if (npart_incoming_xminymax > 0) {
            xminymax_incoming_indices = (npy_intp*)malloc(npart_incoming_xminymax * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_xminymax; i++) xminymax_incoming_indices[i] = 0;
        }
        if (npart_incoming_xmaxymax > 0) {
            xmaxymax_incoming_indices = (npy_intp*)malloc(npart_incoming_xmaxymax * sizeof(npy_intp));
            for (npy_intp i = 0; i < npart_incoming_xmaxymax; i++) xmaxymax_incoming_indices[i] = 0;
        }
        
        // Get indices of incoming particles
        get_incoming_index(
            x_list, y_list, npart_list,
            xmin_list, xmax_list, ymin_list, ymax_list,
            // boundary indices
            xmin_index, xmax_index, ymin_index, ymax_index,
            xminymin_index, xmaxymin_index, xminymax_index, xmaxymax_index,
            // out
            xmin_incoming_indices, xmax_incoming_indices,
            ymin_incoming_indices, ymax_incoming_indices,
            xminymin_incoming_indices, xmaxymin_incoming_indices,
            xminymax_incoming_indices, xmaxymax_incoming_indices
        );
        
        // Allocate buffer for incoming particles with cleanup attribute
        AUTOFREE double* buffer = malloc(nattrs*npart_new * sizeof(double));
        // Fill buffer with boundary particles
        fill_boundary_particles_to_buffer(
            attrs_list, nattrs,
            xmin_incoming_indices, xmax_incoming_indices,
            ymin_incoming_indices, ymax_incoming_indices,
            xminymin_incoming_indices, xmaxymin_incoming_indices,
            xminymax_incoming_indices, xmaxymax_incoming_indices,
            npart_incoming_xmin, npart_incoming_xmax,
            npart_incoming_ymin, npart_incoming_ymax,
            npart_incoming_xminymin, npart_incoming_xmaxymin,
            npart_incoming_xminymax, npart_incoming_xmaxymax,
            xmin_index, xmax_index, ymin_index, ymax_index,
            xminymin_index, xmaxymin_index, xminymax_index, xmaxymax_index,
            buffer
        );
        
        // Fill particles from buffer
        npy_intp ibuff = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if (ibuff >= npart_new) {
                break;
            }
            if (is_dead[ipart]) {
                for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
                    attrs_list[ipatch*nattrs + iattr][ipart] = buffer[ibuff*nattrs+iattr];
                }
                is_dead[ipart] = 0; // Mark as alive
                ibuff++;
            }
        }

        mark_out_of_bound_as_dead(
            x_list[ipatch], y_list[ipatch], is_dead_list[ipatch], npart_list[ipatch],
            xmin_list[ipatch], xmax_list[ipatch],
            ymin_list[ipatch], ymax_list[ipatch],
            npatches
        );
        
    }
    NPY_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// Module method definitions
static PyMethodDef SyncParticlesMethods[] = {
    {"get_npart_to_extend", get_npart_to_extend, METH_VARARGS, "count the number of particles to be extended, and return the number of new particles"},
    {"fill_particles_from_boundary", fill_particles_from_boundary, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef syncparticlesmodule = {
    PyModuleDef_HEAD_INIT,
    "sync_particles",
    NULL,
    -1,
    SyncParticlesMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_sync_particles(void) {
    import_array();
    return PyModule_Create(&syncparticlesmodule);
}
