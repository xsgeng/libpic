#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include "../utils/cutils.h"

#undef INDEX2
#undef INDEX3

#define INDEX2(i, j) \
    ((j) >= 0 ? (j) : (j) + (NY)) + \
    ((i) >= 0 ? (i) : (i) + (NX)) * (NY)

#define INDEX3(i, j, k) \
    ((k) >= 0 ? (k) : (k) + (NZ)) + \
    ((j) >= 0 ? (j) : (j) + (NY)) * (NZ) + \
    ((i) >= 0 ? (i) : (i) + (NX)) * (NY) * (NZ)


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


static PyObject* sync_currents_2d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list;
    npy_intp npatches, nx, ny, ng;

    if (!PyArg_ParseTuple(args, "OOnnnn", 
        &fields_list, &patches_list,
        &npatches, &nx, &ny, &ng)) {
        return NULL;
    }
    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    double **jx = get_attr_array_double(fields_list, npatches, "jx");
    double **jy = get_attr_array_double(fields_list, npatches, "jy"); 
    double **jz = get_attr_array_double(fields_list, npatches, "jz");
    double **rho = get_attr_array_double(fields_list, npatches, "rho");

    npy_intp **neighbor_index_list = get_attr_array_int(patches_list, npatches, "neighbor_index");

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*4; i++) {
        int field_type = i % 4; // 0:jx, 1:jy, 2:jz, 3:rho
        npy_intp ipatch = i / 4;
        
        double** field = NULL;
        switch(field_type) {
            case 0: field = jx; break;
            case 1: field = jy; break;
            case 2: field = jz; break;
            case 3: field = rho; break;
        }

        #define SYNC_BOUNDARY_2D(src_patch, \
            ix_dst, ix_src, NX, \
            iy_dst, iy_src, NY) \
        for (npy_intp ix = 0; ix < NX; ix++) { \
            for (npy_intp iy = 0; iy < NY; iy++) { \
                field[ipatch][INDEX2(ix_dst+ix, iy_dst+iy)] += field[src_patch][INDEX2(ix_src+ix, iy_src+iy)]; \
                field[src_patch][INDEX2(ix_src+ix, iy_src+iy)] = 0.0; \
            } \
        }
        
        const npy_intp* neighbor_index = neighbor_index_list[ipatch];
        // X direction sync
        if (neighbor_index[XMIN] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[XMIN], 
                0, nx, ng, 
                0, 0, ny
            )
        }
        
        if (neighbor_index[XMAX] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[XMAX], 
                nx-ng, -ng, ng, 
                0, 0, ny
            )
        }

        // Y direction sync
        if (neighbor_index[YMIN] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[YMIN], 
                0, 0, nx, 
                0, ny, ng
            )
        }
        
        if (neighbor_index[YMAX] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[YMAX], 
                0, 0, nx, 
                ny-ng, -ng, ng
            )
        }

        // Corner synchronization
        if (neighbor_index[XMINYMIN] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[XMINYMIN], 
                0, nx, ng, 
                0, ny, ng
            )
        }
        
        if (neighbor_index[XMAXYMIN] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[XMAXYMIN], 
                nx-ng, -ng, ng, 
                0, ny, ng
            )
        }
        
        if (neighbor_index[XMINYMAX] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[XMINYMAX], 
                0, nx, ng, 
                ny-ng, -ng, ng
            )
        }
        
        if (neighbor_index[XMAXYMAX] >= 0) {
            SYNC_BOUNDARY_2D(neighbor_index[XMAXYMAX], 
                nx-ng, -ng, ng, 
                ny-ng, -ng, ng
            )
        }
    }
    Py_END_ALLOW_THREADS

    // Clean up resources
    free(jx); free(jy); free(jz); free(rho);
    free(neighbor_index_list);
    // Py_DECREF(fields_list);
    // Py_DECREF(patches_list);
    
    Py_RETURN_NONE;
}

static PyObject* sync_guard_fields_2d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list;
    npy_intp npatches, nx, ny, ng;

    if (!PyArg_ParseTuple(args, "OOnnnn", 
        &fields_list, &patches_list,
        &npatches, &nx, &ny, &ng)) {
        return NULL;
    }

    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;

    double **ex = get_attr_array_double(fields_list, npatches, "ex");
    double **ey = get_attr_array_double(fields_list, npatches, "ey");
    double **ez = get_attr_array_double(fields_list, npatches, "ez");
    double **bx = get_attr_array_double(fields_list, npatches, "bx");
    double **by = get_attr_array_double(fields_list, npatches, "by");
    double **bz = get_attr_array_double(fields_list, npatches, "bz");

    npy_intp **neighbor_index_list = get_attr_array_int(patches_list, npatches, "neighbor_index");

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*6; i++) {
        int field_type = i % 6; // 0:ex,1:ey,2:ez,3:bx,4:by,5:bz
        npy_intp ipatch = i / 6;
        
        double** field = NULL;
        switch(field_type) {
            case 0: field = ex; break;
            case 1: field = ey; break;
            case 2: field = ez; break;
            case 3: field = bx; break;
            case 4: field = by; break;
            case 5: field = bz; break;
        }

        #define SYNC_GUARD_2D(src_patch, \
            ix_dst, ix_src, NX, \
            iy_dst, iy_src, NY) \
        for (npy_intp ix = 0; ix < NX; ix++) { \
            for (npy_intp iy = 0; iy < NY; iy++) { \
                field[ipatch][INDEX2(ix_dst+ix, iy_dst+iy)] = field[src_patch][INDEX2(ix_src+ix, iy_src+iy)]; \
            } \
        }

        const npy_intp* neighbor_index = neighbor_index_list[ipatch];

        // X direction sync
        if (neighbor_index[XMIN] >= 0) {
            SYNC_GUARD_2D(neighbor_index[XMIN], 
                -ng, nx-ng, ng, 
                0, 0, ny
            )
        }
        
        if (neighbor_index[XMAX] >= 0) {
            SYNC_GUARD_2D(neighbor_index[XMAX], 
                nx, 0, ng, 
                0, 0, ny
            )
        }

        // Y direction sync
        if (neighbor_index[YMIN] >= 0) {
            SYNC_GUARD_2D(neighbor_index[YMIN], 
                0, 0, nx, 
                -ng, ny-ng, ng
            )
        }
        
        if (neighbor_index[YMAX] >= 0) {
            SYNC_GUARD_2D(neighbor_index[YMAX], 
                0, 0, nx, 
                ny, 0, ng
            )
        }

        // Corner synchronization
        if (neighbor_index[XMINYMIN] >= 0) {
            SYNC_GUARD_2D(neighbor_index[XMINYMIN], 
                -ng, nx-ng, ng, 
                -ng, ny-ng, ng
            )
        }
        
        if (neighbor_index[XMAXYMIN] >= 0) {
            SYNC_GUARD_2D(neighbor_index[XMAXYMIN], 
                nx, 0, ng, 
                -ng, ny-ng, ng
            )
        }
        
        if (neighbor_index[XMINYMAX] >= 0) {
            SYNC_GUARD_2D(neighbor_index[XMINYMAX], 
                -ng, nx-ng, ng, 
                ny, 0, ng
            )
        }
        
        if (neighbor_index[XMAXYMAX] >= 0) {
            SYNC_GUARD_2D(neighbor_index[XMAXYMAX], 
                nx, 0, ng, 
                ny, 0, ng
            )
        }
    }
    Py_END_ALLOW_THREADS

    // Clean up resources
    free(ex); free(ey); free(ez);
    free(bx); free(by); free(bz);
    free(neighbor_index_list);
    // Py_DECREF(fields_list);
    // Py_DECREF(patches_list);
    
    Py_RETURN_NONE;
}


static PyMethodDef Methods[] = {
    {"sync_currents_2d", sync_currents_2d, METH_VARARGS, "Synchronize currents between patches (2D)"},
    {"sync_guard_fields_2d", sync_guard_fields_2d, METH_VARARGS, "Synchronize guard cells between patches (2D)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "sync_fields2d",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_sync_fields2d(void) {
    import_array();
    return PyModule_Create(&module);
}
