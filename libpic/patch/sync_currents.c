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
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    field[ipatch][INDEX(ixg, iy)] += field[xmin_ipatch][INDEX(nx+ixg, iy)];
                    field[xmin_ipatch][INDEX(nx+ixg, iy)] = 0.0;
                }
            }
        }
        
        if (xmax_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    field[ipatch][INDEX(nx-ng+ixg, iy)] += field[xmax_ipatch][INDEX(-ixg, iy)];
                    field[xmax_ipatch][INDEX(-ixg, iy)] = 0.0;
                }
            }
        }

        // Y direction sync
        if (ymin_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    field[ipatch][INDEX(ix, iyg)] += field[ymin_ipatch][INDEX(ix, ny+iyg)];
                    field[ymin_ipatch][INDEX(ix, ny+iyg)] = 0.0;
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    field[ipatch][INDEX(ix, ny-ng+iyg)] += field[ymax_ipatch][INDEX(ix, -iyg)];
                    field[ymax_ipatch][INDEX(ix, -iyg)] = 0.0;
                }
            }
        }

        // Corner synchronization
        if (ymin_ipatch >= 0) {
            npy_intp xminymin_ipatch = xmin_index[ymin_ipatch];
            if (xminymin_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX(ixg, iyg)] += field[xminymin_ipatch][INDEX(nx+ixg, ny+iyg)];
                        field[xminymin_ipatch][INDEX(nx+ixg, ny+iyg)] = 0.0;
                    }
                }
            }
            
            npy_intp xmaxymin_ipatch = xmax_index[ymin_ipatch];
            if (xmaxymin_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX(nx-ng+ixg, iyg)] += field[xmaxymin_ipatch][INDEX(-ixg, ny+iyg)];
                        field[xmaxymin_ipatch][INDEX(-ixg, ny+iyg)] = 0.0;
                    }
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            npy_intp xminymax_ipatch = xmin_index[ymax_ipatch];
            if (xminymax_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX(ixg, ny-ng+iyg)] += field[xminymax_ipatch][INDEX(nx+ixg, -iyg)];
                        field[xminymax_ipatch][INDEX(nx+ixg, -iyg)] = 0.0;
                    }
                }
            }
            
            npy_intp xmaxymax_ipatch = xmax_index[ymax_ipatch];
            if (xmaxymax_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX(nx-ng+ixg, ny-ng+iyg)] += field[xmaxymax_ipatch][INDEX(-ixg, -iyg)];
                        field[xmaxymax_ipatch][INDEX(-ixg, -iyg)] = 0.0;
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
