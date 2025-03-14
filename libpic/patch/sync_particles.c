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

enum Boundary2D {
    XMIN = 0,
    XMAX,
    YMIN,
    YMAX,
    XMINYMIN,
    XMAXYMIN,
    XMINYMAX,
    XMAXYMAX,
    NUM_BOUNDARIES
};

static const enum Boundary2D OPPOSITE_BOUNDARY[NUM_BOUNDARIES] = {
    XMAX,
    XMIN,
    YMAX,
    YMIN,
    XMAXYMAX,
    XMINYMAX,
    XMAXYMIN,
    XMINYMIN
};

// Implementation of count_outgoing_particles function
static void count_outgoing_particles(
    double* x, double* y, 
    double xmin, double xmax, double ymin, double ymax,
    npy_intp npart,
    npy_intp* npart_out
) {
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (y[ip] < ymin) {
            if (x[ip] < xmin) {
                (npart_out[XMINYMIN])++;
                continue;
            }
            else if (x[ip] > xmax) {
                (npart_out[XMAXYMIN])++;
                continue;
            }
            else {
                (npart_out[YMIN])++;
                continue;
            }
        }
        else if (y[ip] > ymax) {
            if (x[ip] < xmin) {
                (npart_out[XMINYMAX])++;
                continue;
            }
            else if (x[ip] > xmax) {
                (npart_out[XMAXYMAX])++;
                continue;
            }
            else {
                (npart_out[YMAX])++;
                continue;
            }
        }
        else {
            if (x[ip] < xmin) {
                (npart_out[XMIN])++;
                continue;
            }
            else if (x[ip] > xmax) {
                (npart_out[XMAX])++;
                continue;
            }
        }
    }
}

// Get indices of incoming particles from neighboring patches
static void get_incoming_index(
    double** x_list, double** y_list, npy_intp* npart_list,
    double* xmin_list, double* xmax_list, double* ymin_list, double* ymax_list,
    npy_intp* boundary_index,
    // out
    npy_intp** incoming_indices
) {
    // On xmin boundary
    if ((boundary_index[XMIN] >= 0) && (incoming_indices[XMIN] != NULL)) {
        double* x_on_xmin = x_list[boundary_index[XMIN]];
        double* y_on_xmin = y_list[boundary_index[XMIN]];
        
        double xmax = xmax_list[boundary_index[XMIN]];
        double ymin = ymin_list[boundary_index[XMIN]];
        double ymax = ymax_list[boundary_index[XMIN]];

        npy_intp npart = npart_list[boundary_index[XMIN]];
        
        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmin[ipart] > xmax) && (y_on_xmin[ipart] >= ymin) && (y_on_xmin[ipart] <= ymax)) {
                incoming_indices[XMIN][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xmax boundary
    if ((boundary_index[XMAX] >= 0) && (incoming_indices[XMAX] != NULL)) {
        double* x_on_xmax = x_list[boundary_index[XMAX]];
        double* y_on_xmax = y_list[boundary_index[XMAX]];
        
        double xmin = xmin_list[boundary_index[XMAX]];
        double ymin = ymin_list[boundary_index[XMAX]];
        double ymax = ymax_list[boundary_index[XMAX]];

        npy_intp npart = npart_list[boundary_index[XMAX]];
        
        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmax[ipart] < xmin) && (y_on_xmax[ipart] >= ymin) && (y_on_xmax[ipart] <= ymax)) {
                incoming_indices[XMAX][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On ymin boundary
    if ((boundary_index[YMIN] >= 0) && (incoming_indices[YMIN] != NULL)) {
        double* x_on_ymin = x_list[boundary_index[YMIN]];
        double* y_on_ymin = y_list[boundary_index[YMIN]];
        
        double xmin = xmin_list[boundary_index[YMIN]];
        double xmax = xmax_list[boundary_index[YMIN]];
        double ymax = ymax_list[boundary_index[YMIN]];

        npy_intp npart = npart_list[boundary_index[YMIN]];
        
        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_ymin[ipart] >= xmin) && (x_on_ymin[ipart] <= xmax) && (y_on_ymin[ipart] > ymax)) {
                incoming_indices[YMIN][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On ymax boundary
    if ((boundary_index[YMAX] >= 0) && (incoming_indices[YMAX] != NULL)) {
        double* x_on_ymax = x_list[boundary_index[YMAX]];
        double* y_on_ymax = y_list[boundary_index[YMAX]];
        
        double xmin = xmin_list[boundary_index[YMAX]];
        double xmax = xmax_list[boundary_index[YMAX]];
        double ymin = ymin_list[boundary_index[YMAX]];

        npy_intp npart = npart_list[boundary_index[YMAX]];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_ymax[ipart] >= xmin) && (x_on_ymax[ipart] <= xmax) && (y_on_ymax[ipart] < ymin)) {
                incoming_indices[YMAX][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xminymin boundary
    if ((boundary_index[XMINYMIN] >= 0) && (incoming_indices[XMINYMIN] != NULL)) {
        double* x_on_xminymin = x_list[boundary_index[XMINYMIN]];
        double* y_on_xminymin = y_list[boundary_index[XMINYMIN]];
        
        double xmax = xmax_list[boundary_index[XMINYMIN]];
        double ymax = ymax_list[boundary_index[XMINYMIN]];

        npy_intp npart = npart_list[boundary_index[XMINYMIN]];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xminymin[ipart] > xmax) && (y_on_xminymin[ipart] > ymax)) {
                incoming_indices[XMINYMIN][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xmaxymin boundary
    if ((boundary_index[XMAXYMIN] >= 0) && (incoming_indices[XMAXYMIN] != NULL)) {
        double* x_on_xmaxymin = x_list[boundary_index[XMAXYMIN]];
        double* y_on_xmaxymin = y_list[boundary_index[XMAXYMIN]];
        
        double xmin = xmin_list[boundary_index[XMAXYMIN]];
        double ymax = ymax_list[boundary_index[XMAXYMIN]];

        npy_intp npart = npart_list[boundary_index[XMAXYMIN]];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmaxymin[ipart] < xmin) && (y_on_xmaxymin[ipart] > ymax)) {
                incoming_indices[XMAXYMIN][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xminymax boundary
    if ((boundary_index[XMINYMAX] >= 0) && (incoming_indices[XMINYMAX] != NULL)) {
        double* x_on_xminymax = x_list[boundary_index[XMINYMAX]];
        double* y_on_xminymax = y_list[boundary_index[XMINYMAX]];
        
        double xmax = xmax_list[boundary_index[XMINYMAX]];
        double ymin = ymin_list[boundary_index[XMINYMAX]];

        npy_intp npart = npart_list[boundary_index[XMINYMAX]];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xminymax[ipart] > xmax) && (y_on_xminymax[ipart] < ymin)) {
                incoming_indices[XMINYMAX][i] = ipart;
                i++;
                if (i >= npart) {
                    break;
                }
            }
        }
    }
    
    // On xmaxymax boundary
    if ((boundary_index[XMAXYMAX] >= 0) && (incoming_indices[XMAXYMAX] != NULL)) {
        double* x_on_xmaxymax = x_list[boundary_index[XMAXYMAX]];
        double* y_on_xmaxymax = y_list[boundary_index[XMAXYMAX]];
        
        double xmin = xmin_list[boundary_index[XMAXYMAX]];
        double ymin = ymin_list[boundary_index[XMAXYMAX]];

        npy_intp npart = npart_list[boundary_index[XMAXYMAX]];

        npy_intp i = 0;
        for (npy_intp ipart = 0; ipart < npart; ipart++) {
            if ((x_on_xmaxymax[ipart] < xmin) && (y_on_xmaxymax[ipart] < ymin)) {
                incoming_indices[XMAXYMAX][i] = ipart;
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
    double *x, double *y, npy_bool *is_dead, npy_intp npart, 
    double xmin, double xmax, 
    double ymin, double ymax
) {
    for (npy_intp ipart = 0; ipart < npart; ipart++) {
        if (is_dead[ipart]) {
            x[ipart] = NAN;
            y[ipart] = NAN;
            continue;
        }
        if ((x[ipart] < xmin) || (x[ipart] > xmax) || (y[ipart] < ymin) || (y[ipart] > ymax)) {
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
        double *x = x_list[ipatch];
        double *y = y_list[ipatch];
        double xmin = xmin_list[ipatch];
        double xmax = xmax_list[ipatch];
        double ymin = ymin_list[ipatch];
        double ymax = ymax_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        
        // Count particles going out of bounds
        npy_intp npart_out[NUM_BOUNDARIES] = {0};
        
        count_outgoing_particles(
            x, y, xmin, xmax, ymin, ymax, npart,
            npart_out
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
            xminymin_index_list[ipatch],
            xmaxymin_index_list[ipatch],
            xminymax_index_list[ipatch],
            xmaxymax_index_list[ipatch]
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

    // Adjust particle boundaries
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
    }
    
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

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_intp npart_new = npart_incoming[ipatch];
        if (npart_new <= 0) {
            continue;
        }
        
        npy_bool* is_dead = is_dead_list[ipatch];
        npy_intp npart = npart_list[ipatch];

        npy_intp boundary_index[NUM_BOUNDARIES] = {
            xmin_index_list[ipatch],
            xmax_index_list[ipatch],
            ymin_index_list[ipatch],
            ymax_index_list[ipatch],
            xminymin_index_list[ipatch],
            xmaxymin_index_list[ipatch],
            xminymax_index_list[ipatch],
            xmaxymax_index_list[ipatch]
        };
        
        // Number of particles coming from each boundary
        npy_intp npart_incoming[NUM_BOUNDARIES];
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npy_intp i = boundary_index[ibound];
            if (i >= 0) {
                npart_incoming[ibound] = npart_outgoing[i*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound]];
            } else {
                npart_incoming[ibound] = 0;
            }
        }

        // Indices of particles coming from boundary
        npy_intp* incoming_indices[NUM_BOUNDARIES] = {NULL};
        
        // Allocate memory for indices
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            if (npart_incoming[ibound] > 0) {
                incoming_indices[ibound] = (npy_intp*)malloc(npart_incoming[ibound] * sizeof(npy_intp));
                for (npy_intp i = 0; i < npart_incoming[ibound]; i++) incoming_indices[ibound][i] = 0;
            }
        }
        
        // Get indices of incoming particles
        get_incoming_index(
            x_list, y_list, npart_list,
            xmin_list, xmax_list, ymin_list, ymax_list,
            // boundary indices
            boundary_index,
            // out
            incoming_indices
        );
        
        // Allocate buffer for incoming particles with cleanup attribute
        double* buffer AUTOFREE = NULL;                                                                                                                                                                                                                                          
        if (npart_new > 0) {                                                                                                                                                                                                                                                     
            buffer = malloc(nattrs*npart_new * sizeof(double));                                                                                                                                                                                                                  
            // Fill buffer with boundary particles
            fill_boundary_particles_to_buffer(
                attrs_list, nattrs,
                incoming_indices,
                npart_incoming,
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
        
    }
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        mark_out_of_bound_as_dead(
            x_list[ipatch], y_list[ipatch], is_dead_list[ipatch], npart_list[ipatch],
            xmin_list[ipatch], xmax_list[ipatch],
            ymin_list[ipatch], ymax_list[ipatch]
        );
    }
    Py_END_ALLOW_THREADS

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
