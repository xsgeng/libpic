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
    PyArrayObject *npart_to_extend_array = PyArray_Zeros(1, &npatches, NPY_INT64, NPY_CARRAY);
    PyArrayObject *npart_incoming_array = PyArray_Zeros(1, &npatches, NPY_INT64, NPY_CARRAY);
    PyArrayObject *npart_outgoing_array = PyArray_Zeros(1, &dims, NPY_INT64, NPY_CARRAY);

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

// Module method definitions
static PyMethodDef SyncParticlesMethods[] = {
    {"get_npart_to_extend", get_npart_to_extend, METH_VARARGS, 
     "count the number of particles to be extended, and return the number of new particles"},
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
