#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define LIGHT_SPEED 299792458.0
#define one_third 0.3333333333333333
#define INDEX(i, j) \
    ((j) >= 0 ? (j) : (j) + (ny)) + \
    ((i) >= 0 ? (i) : (i) + (nx)) * (ny)

#define INDEX3(i, j, k) \
    ((k) >= 0 ? (k) : (k) + (nz)) + \
    ((j) >= 0 ? (j) : (j) + (ny)) * (nz) + \
    ((i) >= 0 ? (i) : (i) + (nx)) * (ny) * (nz)

/*
    Get attribute from a list of objects.
    Returns a pointer to an array of pointers to the attribute of each patch
*/
static inline void** get_attr(
    PyObject* list, 
    npy_intp npatches, 
    size_t type_size, 
    const char* attr
) {
    void **data = malloc(npatches * sizeof(void*));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *npy = PyObject_GetAttrString(PyList_GET_ITEM(list, ipatch), attr);
        data[ipatch] = PyArray_DATA((PyArrayObject*) npy);
        Py_DecRef(npy);
    }
    return data;
}

#define GET_ATTR_DOUBLEARRAY(list, npatches, attr) (double**) get_attr(list, npatches, sizeof(double*), attr)
#define GET_ATTR_BOOLARRAY(list, npatches, attr) (npy_bool**) get_attr(list, npatches, sizeof(npy_bool*), attr)
#define GET_ATTR_DOUBLE(list, npatches, attr) (double*) get_attr(list, npatches, sizeof(double), attr)
#define GET_ATTR_INTP(list, npatches, attr) (npy_intp*) get_attr(list, npatches, sizeof(npy_intp), attr)