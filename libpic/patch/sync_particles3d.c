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

enum Boundary3D {
    // faces
    XMIN = 0,
    XMAX,
    YMIN,
    YMAX,
    ZMIN,
    ZMAX,
    // egdes
    XMINYMIN,
    XMINYMAX,
    XMINZMIN,
    XMINZMAX,
    XMAXYMIN,
    XMAXYMAX,
    XMAXZMIN,
    XMAXZMAX,
    YMINZMIN,
    YMINZMAX,
    YMAXZMIN,
    YMAXZMAX,
    // vertices
    XMINYMINZMIN,
    XMINYMINZMAX,
    XMINYMAXZMIN,
    XMINYMAXZMAX,
    XMAXYMINZMIN,
    XMAXYMINZMAX,
    XMAXYMAXZMIN,
    XMAXYMAXZMAX,
    NUM_BOUNDARIES
};

static const enum Boundary3D OPPOSITE_BOUNDARY[NUM_BOUNDARIES] = {
    // faces
    XMAX,
    XMIN,
    YMAX,
    YMIN,
    ZMAX,
    ZMIN,
    // egdes
    XMAXYMAX,
    XMAXYMIN,
    XMAXZMAX,
    XMAXZMIN,
    XMINYMAX,
    XMINYMIN,
    XMINZMAX,
    XMINZMIN,
    YMAXZMAX,
    YMAXZMIN,
    YMINZMAX,
    YMINZMIN,
    // vertices
    XMAXYMAXZMAX,
    XMAXYMAXZMIN,
    XMAXYMINZMAX,
    XMAXYMINZMIN,
    XMINYMAXZMAX,
    XMINYMAXZMIN,
    XMINYMINZMAX,
    XMINYMINZMIN
};

// Implementation of count_outgoing_particles function
static void count_outgoing_particles(
    double* x, double* y, double* z,
    double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
    npy_intp npart,
    npy_intp* npart_out
) {
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (z[ip] < zmin) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMINZMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMINZMIN])++;
                    continue;
                } else {
                    (npart_out[YMINZMIN])++;
                    continue;
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMAXZMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMAXZMIN])++;
                    continue;
                } else {
                    (npart_out[YMAXZMIN])++;
                    continue;
                }
            } else {
                if (x[ip] < xmin) {
                    (npart_out[XMINZMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXZMIN])++;
                    continue;
                } else {
                    (npart_out[ZMIN])++;
                    continue;
                }
            }
        } else if (z[ip] > zmax) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMINZMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMINZMAX])++;
                    continue;
                } else {
                    (npart_out[YMINZMAX])++;
                    continue;
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMAXZMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMAXZMAX])++;
                    continue;
                } else {
                    (npart_out[YMAXZMAX])++;
                    continue;
                }
            } else {
                if (x[ip] < xmin) {
                    (npart_out[XMINZMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXZMAX])++;
                    continue;
                } else {
                    (npart_out[ZMAX])++;
                    continue;
                }
            }
        } else {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMIN])++;
                    continue;
                } else {
                    (npart_out[YMIN])++;
                    continue;
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMAX])++;
                    continue;
                } else {
                    (npart_out[YMAX])++;
                    continue;
                }
            } else {
                if (x[ip] < xmin) {
                    (npart_out[XMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAX])++;
                    continue;
                } else {
                    continue;
                }
            }
        }
    }
}

#define SET_INCOMING_INDEX(BOUND) \
    ipatch = boundary_index[BOUND]; \
    if (ipatch < 0) continue; \
    ibound = OPPOSITE_BOUNDARY[BOUND]; \
    incoming_indices_list[ipatch][ibound][ibuff[BOUND]] = ip; \
    (ibuff[BOUND])++; \
    continue;

// Get indices of incoming particles from neighboring patches
static void get_incoming_index(
    double* x, double* y, double* z, npy_intp npart,
    double xmin, double xmax, 
    double ymin, double ymax, 
    double zmin, double zmax,
    npy_intp* boundary_index,
    // out
    npy_intp*** incoming_indices_list
) {
    AUTOFREE npy_intp* ibuff = (npy_intp*)malloc(NUM_BOUNDARIES * sizeof(npy_intp));
    for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
        ibuff[ibound] = 0;
    }
    npy_intp ipatch, ibound;
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (z[ip] < zmin) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMINZMIN);
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMINZMIN)
                } else {
                    SET_INCOMING_INDEX(YMINZMIN)
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMAXZMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMAXZMIN)
                } else {
                    SET_INCOMING_INDEX(YMAXZMIN)
                }
            } else {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINZMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXZMIN)
                } else {
                    SET_INCOMING_INDEX(ZMIN)
                }
            }
        } else if (z[ip] > zmax) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMINZMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMINZMAX)
                } else {
                    SET_INCOMING_INDEX(YMINZMAX)
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMAXZMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMAXZMAX)
                } else {
                    SET_INCOMING_INDEX(YMAXZMAX)
                }
            } else {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINZMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXZMAX)
                } else {
                    SET_INCOMING_INDEX(ZMAX)
                }
            }
        } else {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMIN)
                } else {
                    SET_INCOMING_INDEX(YMIN)
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMAX)
                } else {
                    SET_INCOMING_INDEX(YMAX)
                }
            } else {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAX)
                } else {
                }
            }
        }
    }
}

// Fill buffer with boundary particles
static void fill_boundary_particles_to_buffer(
    double** attrs_list, npy_intp nattrs,
    npy_intp** incoming_indices,
    npy_intp* npart_incoming,
    npy_intp* boundary_index,
    double* buffer
) {
    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
        npy_intp ibuff = 0;

        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            if (boundary_index[ibound] >= 0) {
                double* attr = attrs_list[boundary_index[ibound]*nattrs + iattr];
                for (npy_intp i = 0; i < npart_incoming[ibound]; i++) {
                    npy_intp idx = incoming_indices[ibound][i];
                    buffer[ibuff*nattrs+iattr] = attr[idx];
                    ibuff++;
                }
            }
        }
    }
}



// Mark out-of-bound particles as dead
static void mark_out_of_bound_as_dead(
    double *x, double *y, double *z, npy_bool *is_dead, npy_intp npart, 
    double xmin, double xmax, 
    double ymin, double ymax,
    double zmin, double zmax
) {
    for (npy_intp ipart = 0; ipart < npart; ipart++) {
        if (is_dead[ipart]) {
            x[ipart] = NAN;
            y[ipart] = NAN;
            z[ipart] = NAN;
            continue;
        }
        if ((x[ipart] < xmin) || (x[ipart] > xmax) || (y[ipart] < ymin) || (y[ipart] > ymax) || (z[ipart] < zmin) || (z[ipart] > zmax)) {
            is_dead[ipart] = 1;
            x[ipart] = NAN;
            y[ipart] = NAN;
            z[ipart] = NAN;
        }
    }
}

PyObject* get_npart_to_extend(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject* particles_list;
    PyObject* patch_list;
    double dx, dy, dz;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOnddd", 
            &particles_list, 
            &patch_list,
            &npatches,
            &dx, &dy, &dz
        )
    ) {
        return NULL;
    }

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE double **z_list = get_attr_array_double(particles_list, npatches, "z");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");

    AUTOFREE npy_intp **boundary_index_list = (npy_intp**)malloc(npatches * sizeof(npy_intp*));

    // faces
    AUTOFREE npy_intp *xmin_index_list = get_attr_int(patch_list, npatches, "xmin_neighbor_index");
    AUTOFREE npy_intp *xmax_index_list = get_attr_int(patch_list, npatches, "xmax_neighbor_index");
    AUTOFREE npy_intp *ymin_index_list = get_attr_int(patch_list, npatches, "ymin_neighbor_index");
    AUTOFREE npy_intp *ymax_index_list = get_attr_int(patch_list, npatches, "ymax_neighbor_index");
    AUTOFREE npy_intp *zmin_index_list = get_attr_int(patch_list, npatches, "zmin_neighbor_index");
    AUTOFREE npy_intp *zmax_index_list = get_attr_int(patch_list, npatches, "zmax_neighbor_index");
    // egdes
    AUTOFREE npy_intp *xminymin_index_list = get_attr_int(patch_list, npatches, "xminymin_neighbor_index");
    AUTOFREE npy_intp *xminymax_index_list = get_attr_int(patch_list, npatches, "xminymax_neighbor_index");
    AUTOFREE npy_intp *xminzmin_index_list = get_attr_int(patch_list, npatches, "xminzmin_neighbor_index");
    AUTOFREE npy_intp *xminzmax_index_list = get_attr_int(patch_list, npatches, "xminzmax_neighbor_index");
    AUTOFREE npy_intp *xmaxymin_index_list = get_attr_int(patch_list, npatches, "xmaxymin_neighbor_index");
    AUTOFREE npy_intp *xmaxymax_index_list = get_attr_int(patch_list, npatches, "xmaxymax_neighbor_index");
    AUTOFREE npy_intp *xmaxzmin_index_list = get_attr_int(patch_list, npatches, "xmaxzmin_neighbor_index");
    AUTOFREE npy_intp *xmaxzmax_index_list = get_attr_int(patch_list, npatches, "xmaxzmax_neighbor_index");
    AUTOFREE npy_intp *yminzmin_index_list = get_attr_int(patch_list, npatches, "yminzmin_neighbor_index");
    AUTOFREE npy_intp *yminzmax_index_list = get_attr_int(patch_list, npatches, "yminzmax_neighbor_index");
    AUTOFREE npy_intp *ymaxzmin_index_list = get_attr_int(patch_list, npatches, "ymaxzmin_neighbor_index");
    AUTOFREE npy_intp *ymaxzmax_index_list = get_attr_int(patch_list, npatches, "ymaxzmax_neighbor_index");
    // vertices
    AUTOFREE npy_intp *xminyminzmin_index_list = get_attr_int(patch_list, npatches, "xminyminzmin_neighbor_index");
    AUTOFREE npy_intp *xminyminzmax_index_list = get_attr_int(patch_list, npatches, "xminyminzmax_neighbor_index");
    AUTOFREE npy_intp *xminymaxzmin_index_list = get_attr_int(patch_list, npatches, "xminymaxzmin_neighbor_index");
    AUTOFREE npy_intp *xminymaxzmax_index_list = get_attr_int(patch_list, npatches, "xminymaxzmax_neighbor_index");
    AUTOFREE npy_intp *xmaxyminzmin_index_list = get_attr_int(patch_list, npatches, "xmaxyminzmin_neighbor_index");
    AUTOFREE npy_intp *xmaxyminzmax_index_list = get_attr_int(patch_list, npatches, "xmaxyminzmax_neighbor_index");
    AUTOFREE npy_intp *xmaxymaxzmin_index_list = get_attr_int(patch_list, npatches, "xmaxymaxzmin_neighbor_index");
    AUTOFREE npy_intp *xmaxymaxzmax_index_list = get_attr_int(patch_list, npatches, "xmaxymaxzmax_neighbor_index");


    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");
    AUTOFREE double *zmin_list = get_attr_double(patch_list, npatches, "zmin");
    AUTOFREE double *zmax_list = get_attr_double(patch_list, npatches, "zmax");

    // Adjust particle boundaries
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
        zmin_list[ipatch] -= 0.5 * dz;
        zmax_list[ipatch] += 0.5 * dz;
    }

    // Allocate arrays for particle counts with cleanup attributes
    npy_intp dims = NUM_BOUNDARIES*npatches;
    PyArrayObject *npart_to_extend_array = (PyArrayObject*) PyArray_ZEROS(1, &npatches, NPY_INT64, 0);
    PyArrayObject *npart_incoming_array = (PyArrayObject*) PyArray_ZEROS(1, &npatches, NPY_INT64, 0);
    PyArrayObject *npart_outgoing_array = (PyArrayObject*) PyArray_ZEROS(1, &dims, NPY_INT64, 0);

    npy_intp *npart_to_extend = (npy_intp*) PyArray_DATA(npart_to_extend_array);
    npy_intp *npart_incoming = (npy_intp*) PyArray_DATA(npart_incoming_array);
    npy_intp *npart_outgoing = (npy_intp*) PyArray_DATA(npart_outgoing_array);

    // Count outgoing particles for each patch
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        // Count particles going out of bounds
        npy_intp npart_out[NUM_BOUNDARIES] = {0};
        
        count_outgoing_particles(
            x_list[ipatch], y_list[ipatch], z_list[ipatch], 
            xmin_list[ipatch], xmax_list[ipatch], 
            ymin_list[ipatch], ymax_list[ipatch], 
            zmin_list[ipatch], zmax_list[ipatch],
            npart_list[ipatch], npart_out
        );
        
        // Store results in the outgoing array
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npart_outgoing[ipatch * NUM_BOUNDARIES + ibound] = npart_out[ibound];
        }
    }
    
    // Calculate incoming particles for each patch
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_bool *is_dead = is_dead_list[ipatch];
        npy_intp npart = npart_list[ipatch];

        npy_intp boundary_index[NUM_BOUNDARIES] = {
            xmin_index_list[ipatch],
            xmax_index_list[ipatch],
            ymin_index_list[ipatch],
            ymax_index_list[ipatch],
            zmin_index_list[ipatch],
            zmax_index_list[ipatch],
            xminymin_index_list[ipatch],
            xminymax_index_list[ipatch],
            xminzmin_index_list[ipatch],
            xminzmax_index_list[ipatch],
            xmaxymin_index_list[ipatch],
            xmaxymax_index_list[ipatch],
            xmaxzmin_index_list[ipatch],
            xmaxzmax_index_list[ipatch],
            yminzmin_index_list[ipatch],
            yminzmax_index_list[ipatch],
            ymaxzmin_index_list[ipatch],
            ymaxzmax_index_list[ipatch],
            xminyminzmin_index_list[ipatch],
            xminyminzmax_index_list[ipatch],
            xminymaxzmin_index_list[ipatch],
            xminymaxzmax_index_list[ipatch],
            xmaxyminzmin_index_list[ipatch],
            xmaxyminzmax_index_list[ipatch],
            xmaxymaxzmin_index_list[ipatch],
            xmaxymaxzmax_index_list[ipatch]
        };
        
        // Count incoming particles
        npy_intp npart_new = 0;
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npy_intp i = boundary_index[ibound];
            if (i >= 0) {
                npart_new += npart_outgoing[i*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound]];
            }
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
    PyObject *particles_list, *patch_list, *attrs;
    PyArrayObject *npart_incoming_array, *npart_outgoing_array;
    double dx, dy, dz;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOOOndddO", 
            &particles_list, 
            &patch_list,
            &npart_incoming_array,
            &npart_outgoing_array,
            &npatches,
            &dx, &dy, &dz, &attrs
        )
    ) {
        return NULL;
    }

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE double **z_list = get_attr_array_double(particles_list, npatches, "z");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");

    // faces
    AUTOFREE npy_intp *xmin_index_list = get_attr_int(patch_list, npatches, "xmin_neighbor_index");
    AUTOFREE npy_intp *xmax_index_list = get_attr_int(patch_list, npatches, "xmax_neighbor_index");
    AUTOFREE npy_intp *ymin_index_list = get_attr_int(patch_list, npatches, "ymin_neighbor_index");
    AUTOFREE npy_intp *ymax_index_list = get_attr_int(patch_list, npatches, "ymax_neighbor_index");
    AUTOFREE npy_intp *zmin_index_list = get_attr_int(patch_list, npatches, "zmin_neighbor_index");
    AUTOFREE npy_intp *zmax_index_list = get_attr_int(patch_list, npatches, "zmax_neighbor_index");
    // egdes
    AUTOFREE npy_intp *xminymin_index_list = get_attr_int(patch_list, npatches, "xminymin_neighbor_index");
    AUTOFREE npy_intp *xminymax_index_list = get_attr_int(patch_list, npatches, "xminymax_neighbor_index");
    AUTOFREE npy_intp *xminzmin_index_list = get_attr_int(patch_list, npatches, "xminzmin_neighbor_index");
    AUTOFREE npy_intp *xminzmax_index_list = get_attr_int(patch_list, npatches, "xminzmax_neighbor_index");
    AUTOFREE npy_intp *xmaxymin_index_list = get_attr_int(patch_list, npatches, "xmaxymin_neighbor_index");
    AUTOFREE npy_intp *xmaxymax_index_list = get_attr_int(patch_list, npatches, "xmaxymax_neighbor_index");
    AUTOFREE npy_intp *xmaxzmin_index_list = get_attr_int(patch_list, npatches, "xmaxzmin_neighbor_index");
    AUTOFREE npy_intp *xmaxzmax_index_list = get_attr_int(patch_list, npatches, "xmaxzmax_neighbor_index");
    AUTOFREE npy_intp *yminzmin_index_list = get_attr_int(patch_list, npatches, "yminzmin_neighbor_index");
    AUTOFREE npy_intp *yminzmax_index_list = get_attr_int(patch_list, npatches, "yminzmax_neighbor_index");
    AUTOFREE npy_intp *ymaxzmin_index_list = get_attr_int(patch_list, npatches, "ymaxzmin_neighbor_index");
    AUTOFREE npy_intp *ymaxzmax_index_list = get_attr_int(patch_list, npatches, "ymaxzmax_neighbor_index");
    // vertices
    AUTOFREE npy_intp *xminyminzmin_index_list = get_attr_int(patch_list, npatches, "xminyminzmin_neighbor_index");
    AUTOFREE npy_intp *xminyminzmax_index_list = get_attr_int(patch_list, npatches, "xminyminzmax_neighbor_index");
    AUTOFREE npy_intp *xminymaxzmin_index_list = get_attr_int(patch_list, npatches, "xminymaxzmin_neighbor_index");
    AUTOFREE npy_intp *xminymaxzmax_index_list = get_attr_int(patch_list, npatches, "xminymaxzmax_neighbor_index");
    AUTOFREE npy_intp *xmaxyminzmin_index_list = get_attr_int(patch_list, npatches, "xmaxyminzmin_neighbor_index");
    AUTOFREE npy_intp *xmaxyminzmax_index_list = get_attr_int(patch_list, npatches, "xmaxyminzmax_neighbor_index");
    AUTOFREE npy_intp *xmaxymaxzmin_index_list = get_attr_int(patch_list, npatches, "xmaxymaxzmin_neighbor_index");
    AUTOFREE npy_intp *xmaxymaxzmax_index_list = get_attr_int(patch_list, npatches, "xmaxymaxzmax_neighbor_index");


    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");
    AUTOFREE double *zmin_list = get_attr_double(patch_list, npatches, "zmin");
    AUTOFREE double *zmax_list = get_attr_double(patch_list, npatches, "zmax");

    // Adjust particle boundaries
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
        zmin_list[ipatch] -= 0.5 * dz;
        zmax_list[ipatch] += 0.5 * dz;
    }

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
    // Number of particles coming from each boundary
    AUTOFREE npy_intp** npart_incoming_boundary_list = (npy_intp**)malloc(npatches * sizeof(npy_intp**));
    // Indices of particles coming from boundary
    AUTOFREE npy_intp*** incoming_indices_list = (npy_intp***)malloc(npatches * sizeof(npy_intp**));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npart_incoming_boundary_list[ipatch] = (npy_intp*)malloc(NUM_BOUNDARIES * sizeof(npy_intp*));
        incoming_indices_list[ipatch] = (npy_intp**)malloc(NUM_BOUNDARIES * sizeof(npy_intp*));
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            incoming_indices_list[ipatch][ibound] = NULL;
            incoming_indices_list[ipatch][ibound] = 0;
        }
    }

    int num_threads = omp_get_max_threads();
    if (npatches < num_threads) num_threads = npatches;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel num_threads(num_threads)
    {    // Number of particles coming from each boundary
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            npy_intp npart_new = npart_incoming[ipatch];
            if (npart_new <= 0) {
                continue;
            }
            npy_intp boundary_index[NUM_BOUNDARIES] = {
                xmin_index_list[ipatch],
                xmax_index_list[ipatch],
                ymin_index_list[ipatch],
                ymax_index_list[ipatch],
                zmin_index_list[ipatch],
                zmax_index_list[ipatch],
                xminymin_index_list[ipatch],
                xminymax_index_list[ipatch],
                xminzmin_index_list[ipatch],
                xminzmax_index_list[ipatch],
                xmaxymin_index_list[ipatch],
                xmaxymax_index_list[ipatch],
                xmaxzmin_index_list[ipatch],
                xmaxzmax_index_list[ipatch],
                yminzmin_index_list[ipatch],
                yminzmax_index_list[ipatch],
                ymaxzmin_index_list[ipatch],
                ymaxzmax_index_list[ipatch],
                xminyminzmin_index_list[ipatch],
                xminyminzmax_index_list[ipatch],
                xminymaxzmin_index_list[ipatch],
                xminymaxzmax_index_list[ipatch],
                xmaxyminzmin_index_list[ipatch],
                xmaxyminzmax_index_list[ipatch],
                xmaxymaxzmin_index_list[ipatch],
                xmaxymaxzmax_index_list[ipatch]
            };
            
            for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
                npy_intp i = boundary_index[ibound];
                if (i >= 0) {
                    npart_incoming_boundary_list[ipatch][ibound] = npart_outgoing[i*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound]];
                    if (npart_incoming_boundary_list[ipatch][ibound] > 0) {
                        incoming_indices_list[ipatch][ibound] = (npy_intp*)malloc(npart_incoming_boundary_list[ipatch][ibound] * sizeof(npy_intp));
                    }
                }
            }
        }
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            npy_intp boundary_index[NUM_BOUNDARIES] = {
                xmin_index_list[ipatch],
                xmax_index_list[ipatch],
                ymin_index_list[ipatch],
                ymax_index_list[ipatch],
                zmin_index_list[ipatch],
                zmax_index_list[ipatch],
                xminymin_index_list[ipatch],
                xminymax_index_list[ipatch],
                xminzmin_index_list[ipatch],
                xminzmax_index_list[ipatch],
                xmaxymin_index_list[ipatch],
                xmaxymax_index_list[ipatch],
                xmaxzmin_index_list[ipatch],
                xmaxzmax_index_list[ipatch],
                yminzmin_index_list[ipatch],
                yminzmax_index_list[ipatch],
                ymaxzmin_index_list[ipatch],
                ymaxzmax_index_list[ipatch],
                xminyminzmin_index_list[ipatch],
                xminyminzmax_index_list[ipatch],
                xminymaxzmin_index_list[ipatch],
                xminymaxzmax_index_list[ipatch],
                xmaxyminzmin_index_list[ipatch],
                xmaxyminzmax_index_list[ipatch],
                xmaxymaxzmin_index_list[ipatch],
                xmaxymaxzmax_index_list[ipatch]
            };
            
            // Get indices of incoming particles
            get_incoming_index(
                x_list[ipatch], y_list[ipatch], z_list[ipatch], npart_list[ipatch],
                xmin_list[ipatch], xmax_list[ipatch], 
                ymin_list[ipatch], ymax_list[ipatch],
                zmin_list[ipatch], zmax_list[ipatch],
                // boundary indices
                boundary_index,
                // out
                incoming_indices_list
            );
        }
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            npy_intp npart_new = npart_incoming[ipatch];
            if (npart_new <= 0) {
                continue;
            }
            npy_intp boundary_index[NUM_BOUNDARIES] = {
                xmin_index_list[ipatch],
                xmax_index_list[ipatch],
                ymin_index_list[ipatch],
                ymax_index_list[ipatch],
                zmin_index_list[ipatch],
                zmax_index_list[ipatch],
                xminymin_index_list[ipatch],
                xminymax_index_list[ipatch],
                xminzmin_index_list[ipatch],
                xminzmax_index_list[ipatch],
                xmaxymin_index_list[ipatch],
                xmaxymax_index_list[ipatch],
                xmaxzmin_index_list[ipatch],
                xmaxzmax_index_list[ipatch],
                yminzmin_index_list[ipatch],
                yminzmax_index_list[ipatch],
                ymaxzmin_index_list[ipatch],
                ymaxzmax_index_list[ipatch],
                xminyminzmin_index_list[ipatch],
                xminyminzmax_index_list[ipatch],
                xminymaxzmin_index_list[ipatch],
                xminymaxzmax_index_list[ipatch],
                xmaxyminzmin_index_list[ipatch],
                xmaxyminzmax_index_list[ipatch],
                xmaxymaxzmin_index_list[ipatch],
                xmaxymaxzmax_index_list[ipatch]
            };
            npy_bool* is_dead = is_dead_list[ipatch];
            npy_intp npart = npart_list[ipatch];
            // Allocate buffer for incoming particles with cleanup attribute
            AUTOFREE double* buffer = malloc(nattrs*npart_new * sizeof(double));                                                                                                                                                                                                                  
            // Fill buffer with boundary particles
            fill_boundary_particles_to_buffer(
                attrs_list, nattrs,
                incoming_indices_list[ipatch],
                npart_incoming_boundary_list[ipatch],
                boundary_index,
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
        }
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            mark_out_of_bound_as_dead(
                x_list[ipatch], y_list[ipatch], z_list[ipatch], is_dead_list[ipatch], npart_list[ipatch],
                xmin_list[ipatch], xmax_list[ipatch],
                ymin_list[ipatch], ymax_list[ipatch],
                zmin_list[ipatch], zmax_list[ipatch]
            );
        }
    }
    Py_END_ALLOW_THREADS

    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            free(incoming_indices_list[ipatch][ibound]);
        }
        free(npart_incoming_boundary_list[ipatch]);
        free(incoming_indices_list[ipatch]);
    }

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
    "sync_particles3d",
    NULL,
    -1,
    SyncParticlesMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_sync_particles(void) {
    import_array();
    return PyModule_Create(&syncparticlesmodule);
}
