#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define INDEX(i, j, nx, ny) \
    ((j) >= 0 ? (j) : (j) + (ny)) + \
    ((i) >= 0 ? (i) : (i) + (nx)) * (ny)



inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

/**
 * adapted from https://github.com/Warwick-Plasma/epoch/blob/main/epoch2d/src/include/triangle/e_part.inc
 * and https://github.com/Warwick-Plasma/epoch/blob/main/epoch2d/src/include/triangle/b_part.inc
 */
inline static double interp_ex(double* ex, double* hx, double* gy, npy_intp ix2, npy_intp iy1, npy_intp nx, npy_intp ny) {
    double ex_part = 
          gy[0] * (hx[0] * ex[INDEX(ix2-1, iy1-1, nx, ny)] 
        +          hx[1] * ex[INDEX(ix2,   iy1-1, nx, ny)] 
        +          hx[2] * ex[INDEX(ix2+1, iy1-1, nx, ny)])
        + gy[1] * (hx[0] * ex[INDEX(ix2-1, iy1,   nx, ny)] 
        +          hx[1] * ex[INDEX(ix2,   iy1,   nx, ny)] 
        +          hx[2] * ex[INDEX(ix2+1, iy1,   nx, ny)]) 
        + gy[2] * (hx[0] * ex[INDEX(ix2-1, iy1+1, nx, ny)] 
        +          hx[1] * ex[INDEX(ix2,   iy1+1, nx, ny)] 
        +          hx[2] * ex[INDEX(ix2+1, iy1+1, nx, ny)]);
    return ex_part;
}

inline static double interp_ey(double* ey, double* gx, double* hy, npy_intp ix1, npy_intp iy2, npy_intp nx, npy_intp ny) {
    double ey_part =
          hy[ 0] * (gx[ 0] * ey[INDEX(ix1-1,iy2-1, nx, ny)]
        +           gx[ 1] * ey[INDEX(ix1  ,iy2-1, nx, ny)]
        +           gx[ 2] * ey[INDEX(ix1+1,iy2-1, nx, ny)])
        + hy[ 1] * (gx[ 0] * ey[INDEX(ix1-1,iy2  , nx, ny)]
        +           gx[ 1] * ey[INDEX(ix1  ,iy2  , nx, ny)]
        +           gx[ 2] * ey[INDEX(ix1+1,iy2  , nx, ny)])
        + hy[ 2] * (gx[ 0] * ey[INDEX(ix1-1,iy2+1, nx, ny)]
        +           gx[ 1] * ey[INDEX(ix1  ,iy2+1, nx, ny)]
        +           gx[ 2] * ey[INDEX(ix1+1,iy2+1, nx, ny)]);
    return ey_part;
}

inline static double interp_ez(double* ez, double* gx, double* gy, npy_intp ix1, npy_intp iy1, npy_intp nx, npy_intp ny) {
    double ez_part =
          gy[ 0] * (gx[ 0] * ez[INDEX(ix1-1, iy1-1, nx, ny)]
        +           gx[ 1] * ez[INDEX(ix1  , iy1-1, nx, ny)]
        +           gx[ 2] * ez[INDEX(ix1+1, iy1-1, nx, ny)])
        + gy[ 1] * (gx[ 0] * ez[INDEX(ix1-1, iy1  , nx, ny)]
        +           gx[ 1] * ez[INDEX(ix1  , iy1  , nx, ny)]
        +           gx[ 2] * ez[INDEX(ix1+1, iy1  , nx, ny)])
        + gy[ 2] * (gx[ 0] * ez[INDEX(ix1-1, iy1+1, nx, ny)]
        +           gx[ 1] * ez[INDEX(ix1  , iy1+1, nx, ny)]
        +           gx[ 2] * ez[INDEX(ix1+1, iy1+1, nx, ny)]);
    return ez_part;
}

inline static double interp_bx(double* bx, double* gx, double* hy, npy_intp ix1, npy_intp iy2, npy_intp nx, npy_intp ny) {
    double bx_part =
          hy[ 0] * (gx[ 0] * bx[INDEX(ix1-1, iy2-1, nx, ny)]
        +           gx[ 1] * bx[INDEX(ix1  , iy2-1, nx, ny)]
        +           gx[ 2] * bx[INDEX(ix1+1, iy2-1, nx, ny)])
        + hy[ 1] * (gx[ 0] * bx[INDEX(ix1-1, iy2  , nx, ny)]
        +           gx[ 1] * bx[INDEX(ix1  , iy2  , nx, ny)]
        +           gx[ 2] * bx[INDEX(ix1+1, iy2  , nx, ny)])
        + hy[ 2] * (gx[ 0] * bx[INDEX(ix1-1, iy2+1, nx, ny)]
        +           gx[ 1] * bx[INDEX(ix1  , iy2+1, nx, ny)]
        +           gx[ 2] * bx[INDEX(ix1+1, iy2+1, nx, ny)]);
    return bx_part;
}

inline static double interp_by(double* by, double* hx, double* gy, npy_intp ix2, npy_intp iy1, npy_intp nx, npy_intp ny) {
    double by_part = 
          by_part =
          gy[ 0] * (hx[ 0] * by[INDEX(ix2-1, iy1-1, nx, ny)]
        +           hx[ 1] * by[INDEX(ix2  , iy1-1, nx, ny)]
        +           hx[ 2] * by[INDEX(ix2+1, iy1-1, nx, ny)])
        + gy[ 1] * (hx[ 0] * by[INDEX(ix2-1, iy1  , nx, ny)]
        +           hx[ 1] * by[INDEX(ix2  , iy1  , nx, ny)]
        +           hx[ 2] * by[INDEX(ix2+1, iy1  , nx, ny)])
        + gy[ 2] * (hx[ 0] * by[INDEX(ix2-1, iy1+1, nx, ny)]
        +           hx[ 1] * by[INDEX(ix2  , iy1+1, nx, ny)]
        +           hx[ 2] * by[INDEX(ix2+1, iy1+1, nx, ny)]);
    return by_part;
}

inline static double interp_bz(double* bz, double* hx, double* hy, npy_intp ix2, npy_intp iy2, npy_intp nx, npy_intp ny) {
    double bz_part =
          hy[ 0] * (hx[ 0] * bz[INDEX(ix2-1, iy2-1, nx, ny)]
        +           hx[ 1] * bz[INDEX(ix2  , iy2-1, nx, ny)]
        +           hx[ 2] * bz[INDEX(ix2+1, iy2-1, nx, ny)])
        + hy[ 1] * (hx[ 0] * bz[INDEX(ix2-1, iy2  , nx, ny)]
        +           hx[ 1] * bz[INDEX(ix2  , iy2  , nx, ny)]
        +           hx[ 2] * bz[INDEX(ix2+1, iy2  , nx, ny)])
        + hy[ 2] * (hx[ 0] * bz[INDEX(ix2-1, iy2+1, nx, ny)]
        +           hx[ 1] * bz[INDEX(ix2  , iy2+1, nx, ny)]
        +           hx[ 2] * bz[INDEX(ix2+1, iy2+1, nx, ny)]);
    return bz_part;
}

static void interpolation_2d(
    double* x, double* y, 
    double* ex_part, double* ey_part, double* ez_part, 
    double* bx_part, double* by_part, double* bz_part, 
    npy_bool* is_dead,
    npy_intp npart, 
    double* ex, double* ey, double* ez, 
    double* bx, double* by, double* bz, 
    double dx, double dy, double x0, double y0, 
    npy_intp nx, npy_intp ny) {
    
    double gx[3];
    double gy[3];
    double hx[3];
    double hy[3];

    for (npy_intp ip = 0; ip < npart; ip++) {
        if (is_dead[ip]) {
            continue;
        }
        // print if x or y is nan
        if (isnan(x[ip]) || isnan(y[ip])) {
            printf("x or y is nan but not dead.\n");
            continue;
        }
        
        double x_over_dx = (x[ip] - x0) / dx;
        double y_over_dy = (y[ip] - y0) / dy;

        npy_intp ix1 = (int)floor(x_over_dx + 0.5);
        get_gx(ix1 - x_over_dx, gx);

        npy_intp ix2 = (int)floor(x_over_dx);
        get_gx(ix2 - x_over_dx + 0.5, hx);

        npy_intp iy1 = (int)floor(y_over_dy + 0.5);
        get_gx(iy1 - y_over_dy, gy);

        npy_intp iy2 = (int)floor(y_over_dy);
        get_gx(iy2 - y_over_dy + 0.5, hy);

        ex_part[ip] = interp_ex(ex, hx, gy, ix2, iy1, nx, ny);
        ey_part[ip] = interp_ey(ey, gx, hy, ix1, iy2, nx, ny);
        ez_part[ip] = interp_ez(ez, gx, gy, ix1, iy1, nx, ny);

        bx_part[ip] = interp_bx(bx, gx, hy, ix1, iy2, nx, ny);
        by_part[ip] = interp_by(by, hx, gy, ix2, iy1, nx, ny);
        bz_part[ip] = interp_bz(bz, hx, hy, ix2, iy2, nx, ny);
    }
}

/**
 * Python wrapper for the interpolation_2d function.
 */
PyObject* _interpolation_2d(PyObject* self, PyObject* args) {
    PyArrayObject *x, *y, 
                *ex_part, *ey_part, *ez_part, 
                *bx_part, *by_part, *bz_part, 
                *is_dead,
                *ex, *ey, *ez, 
                *bx, *by, *bz;
    double dx, dy, x0, y0;
    npy_intp npart, nx, ny;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOnOOOOOOddddnn", 
        &x, &y, 
        &ex_part, &ey_part, &ez_part, 
        &bx_part, &by_part, &bz_part, 
        &is_dead,
        &npart, 
        &ex, &ey, &ez, 
        &bx, &by, &bz, 
        &dx, &dy, &x0, &y0, 
        &nx, &ny)) {
        return NULL;
    }

    interpolation_2d(
        (double*)PyArray_DATA(x), 
        (double*)PyArray_DATA(y), 
        (double*)PyArray_DATA(ex_part), 
        (double*)PyArray_DATA(ey_part), 
        (double*)PyArray_DATA(ez_part), 
        (double*)PyArray_DATA(bx_part), 
        (double*)PyArray_DATA(by_part), 
        (double*)PyArray_DATA(bz_part), 
        (npy_bool*)PyArray_DATA(is_dead), 
        npart, 
        (double*)PyArray_DATA(ex), 
        (double*)PyArray_DATA(ey), 
        (double*)PyArray_DATA(ez), 
        (double*)PyArray_DATA(bx), 
        (double*)PyArray_DATA(by), 
        (double*)PyArray_DATA(bz), 
        dx, dy, x0, y0, 
        nx, ny);

    Py_RETURN_NONE;
}

static PyObject* interpolation_patches_2d(PyObject* self, PyObject* args) {
    PyObject *x_list, *y_list, 
             *ex_part_list, *ey_part_list, *ez_part_list, 
             *bx_part_list, *by_part_list, *bz_part_list, 
             *ex_list, *ey_list, *ez_list, 
             *bx_list, *by_list, *bz_list, 
             *x0_list, *y0_list, 
             *is_dead_list;
    double dx, dy;
    npy_intp npatches, nx, ny;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOnddnn", 
        &x_list, &y_list, 
        &ex_part_list, &ey_part_list, &ez_part_list, 
        &bx_part_list, &by_part_list, &bz_part_list, 
        &is_dead_list, 
        &ex_list, &ey_list, &ez_list, 
        &bx_list, &by_list, &bz_list, 
        &x0_list, &y0_list, 
        &npatches, &dx, &dy, &nx, &ny)) {
        return NULL;
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;
    }

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        double *x          = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(x_list, ipatch));
        double *y          = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(y_list, ipatch));
        double *ex_part    = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(ex_part_list, ipatch));
        double *ey_part    = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(ey_part_list, ipatch));
        double *ez_part    = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(ez_part_list, ipatch));
        double *bx_part    = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(bx_part_list, ipatch));
        double *by_part    = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(by_part_list, ipatch));
        double *bz_part    = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(bz_part_list, ipatch));
        double *ex         = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(ex_list, ipatch));
        double *ey         = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(ey_list, ipatch));
        double *ez         = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(ez_list, ipatch));
        double *bx         = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(bx_list, ipatch));
        double *by         = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(by_list, ipatch));
        double *bz         = (double*) PyArray_DATA((PyArrayObject*)PyList_GetItem(bz_list, ipatch));
        double x0          = PyFloat_AsDouble(PyList_GetItem(x0_list, ipatch));
        double y0          = PyFloat_AsDouble(PyList_GetItem(y0_list, ipatch));
        npy_bool *is_dead  = (npy_bool*) PyArray_DATA((PyArrayObject*)PyList_GetItem(is_dead_list, ipatch));

        npy_intp npart = PyArray_DIM((PyArrayObject*)PyList_GetItem(x_list, ipatch), 0);

        interpolation_2d(
            x, y, 
            ex_part, ey_part, ez_part, 
            bx_part, by_part, bz_part, 
            is_dead,
            npart, 
            ex, ey, ez, 
            bx, by, bz, 
            dx, dy, x0, y0, 
            nx, ny);
    }

    Py_RETURN_NONE;
}

static PyMethodDef InterpolationMethods[] = {
    {"_interpolation_2d", _interpolation_2d, METH_VARARGS, "2D interpolation on CPU"},
    {"interpolation_patches_2d", interpolation_patches_2d, METH_VARARGS, "2D interpolation on CPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef interpolationmodule = {
    PyModuleDef_HEAD_INIT,
    "cpu",
    NULL,
    -1,
    InterpolationMethods
};

PyMODINIT_FUNC PyInit_cpu(void) {
    import_array();
    return PyModule_Create(&interpolationmodule);
}
