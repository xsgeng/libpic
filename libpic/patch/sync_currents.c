#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include "../utils/cutils.h"

static PyObject* sync_currents(PyObject* self, PyObject* args) {
    PyObject *fields_list;
    PyArrayObject *xmin_index_arr, *xmax_index_arr, *ymin_index_arr, *ymax_index_arr;
    npy_intp npatches, nx, ny, ng;

    if (!PyArg_ParseTuple(args, "OOOOOnnnn", 
        &fields_list, 
        &xmin_index_arr, &xmax_index_arr,
        &ymin_index_arr, &ymax_index_arr,
        &npatches, &nx, &ny, &ng)) {
        return NULL;
    }

    // Get field data pointers
    double **jx = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "jx");
    double **jy = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "jy"); 
    double **jz = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "jz");
    double **rho = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "rho");

    // Get boundary index arrays
    npy_intp* xmin_index = (npy_intp*)PyArray_DATA(xmin_index_arr);
    npy_intp* xmax_index = (npy_intp*)PyArray_DATA(xmax_index_arr);
    npy_intp* ymin_index = (npy_intp*)PyArray_DATA(ymin_index_arr);
    npy_intp* ymax_index = (npy_intp*)PyArray_DATA(ymax_index_arr);

    // Release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*4; i++) {
        int field_type = i % 4; // 0:jx, 1:jy, 2:jz, 3:rho
        npy_intp ipatch = i / 4;
        
        double** field;
        switch(field_type) {
            case 0: field = jx; break;
            case 1: field = jy; break;
            case 2: field = jz; break;
            case 3: field = rho; break;
        }

        npy_intp xmin_ipatch = xmin_index[ipatch];
        npy_intp xmax_ipatch = xmax_index[ipatch];
        npy_intp ymin_ipatch = ymin_index[ipatch];
        npy_intp ymax_ipatch = ymax_index[ipatch];

        // X direction sync
        if (xmin_ipatch >= 0) {
            for (npy_intp g = 0; g < ng; g++) {
                for (npy_intp y = 0; y < ny; y++) {
                    field[ipatch][INDEX(g, y)] += field[xmin_ipatch][INDEX(nx+g, y)];
                    field[xmin_ipatch][INDEX(nx+g, y)] = 0.0;
                }
            }
        }
        
        if (xmax_ipatch >= 0) {
            for (npy_intp g = 0; g < ng; g++) {
                for (npy_intp y = 0; y < ny; y++) {
                    field[ipatch][INDEX(nx-ng+g, y)] += field[xmax_ipatch][INDEX(g, y)];
                    field[xmax_ipatch][INDEX(g, y)] = 0.0;
                }
            }
        }

        // Y direction sync
        if (ymin_ipatch >= 0) {
            for (npy_intp x = 0; x < nx; x++) {
                for (npy_intp g = 0; g < ng; g++) {
                    field[ipatch][INDEX(x, g)] += field[ymin_ipatch][INDEX(x, ny+g)];
                    field[ymin_ipatch][INDEX(x, ny+g)] = 0.0;
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            for (npy_intp x = 0; x < nx; x++) {
                for (npy_intp g = 0; g < ng; g++) {
                    field[ipatch][INDEX(x, ny-ng+g)] += field[ymax_ipatch][INDEX(x, g)];
                    field[ymax_ipatch][INDEX(x, g)] = 0.0;
                }
            }
        }

        // Corner synchronization
        if (ymin_ipatch >= 0) {
            npy_intp xminymin_ipatch = xmin_index[ymin_ipatch];
            if (xminymin_ipatch >= 0) {
                for (npy_intp gx = 0; gx < ng; gx++) {
                    for (npy_intp gy = 0; gy < ng; gy++) {
                        field[ipatch][INDEX(gx, gy)] += field[xminymin_ipatch][INDEX(nx+gx, ny+gy)];
                        field[xminymin_ipatch][INDEX(nx+gx, ny+gy)] = 0.0;
                    }
                }
            }
            
            npy_intp xmaxymin_ipatch = xmax_index[ymin_ipatch];
            if (xmaxymin_ipatch >= 0) {
                for (npy_intp gx = 0; gx < ng; gx++) {
                    for (npy_intp gy = 0; gy < ng; gy++) {
                        field[ipatch][INDEX(nx-ng+gx, gy)] += field[xmaxymin_ipatch][INDEX(gx, ny+gy)];
                        field[xmaxymin_ipatch][INDEX(gx, ny+gy)] = 0.0;
                    }
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            npy_intp xminymax_ipatch = xmin_index[ymax_ipatch];
            if (xminymax_ipatch >= 0) {
                for (npy_intp gx = 0; gx < ng; gx++) {
                    for (npy_intp gy = 0; gy < ng; gy++) {
                        field[ipatch][INDEX(gx, ny-ng+gy)] += field[xminymax_ipatch][INDEX(nx+gx, gy)];
                        field[xminymax_ipatch][INDEX(nx+gx, gy)] = 0.0;
                    }
                }
            }
            
            npy_intp xmaxymax_ipatch = xmax_index[ymax_ipatch];
            if (xmaxymax_ipatch >= 0) {
                for (npy_intp gx = 0; gx < ng; gx++) {
                    for (npy_intp gy = 0; gy < ng; gy++) {
                        field[ipatch][INDEX(nx-ng+gx, ny-ng+gy)] += field[xmaxymax_ipatch][INDEX(gx, gy)];
                        field[xmaxymax_ipatch][INDEX(gx, gy)] = 0.0;
                    }
                }
            }
        }
    }
    // Acquire GIL
    Py_END_ALLOW_THREADS

    // Clean up resources
    free(jx); free(jy); free(jz); free(rho);
    Py_DECREF(fields_list);
    
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"sync_currents", sync_currents, METH_VARARGS, "Synchronize currents between patches"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "sync_currents",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_sync_currents(void) {
    import_array();
    return PyModule_Create(&module);
}
