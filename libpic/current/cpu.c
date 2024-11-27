#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define LIGHT_SPEED 299792458.0
#define one_third 0.3333333333333333
#define INDEX(i, j, nx, ny) \
    ((j) >= 0 ? (j) : (j) + (ny)) + \
    ((i) >= 0 ? (i) : (i) + (nx)) * (ny)
#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))

static void calculate_S(double delta, int shift, double* S) {
    double delta2 = delta * delta;

    double delta_minus = 0.5 * (delta2 + delta + 0.25);
    double delta_mid = 0.75 - delta2;
    double delta_positive = 0.5 * (delta2 - delta + 0.25);

    int minus = shift == -1;
    int mid = shift == 0;
    int positive = shift == 1;

    S[0] = minus * delta_minus;
    S[1] = minus * delta_mid + mid * delta_minus;
    S[2] = minus * delta_positive + mid * delta_mid + positive * delta_minus;
    S[3] = mid * delta_positive + positive * delta_mid;
    S[4] = positive * delta_positive;
}

static void current_deposit_2d(
    double* rho, double* jx, double* jy, double* jz, 
    double* x, double* y, double* ux, double* uy, double* uz, double* inv_gamma, 
    npy_bool* is_dead, 
    npy_intp npart, npy_intp nx, npy_intp ny,
    double dx, double dy, double x0, double y0, double dt, double* w, double q
) {
    double x_old, y_old, x_adv, y_adv;
    double S0x[5], S1x[5], S0y[5], S1y[5], DSx[5], DSy[5], jy_buff[5];
    
    npy_intp i, j, ipart;
    
    npy_intp dcell_x, dcell_y, ix0, iy0, ix1, iy1, ix, iy;
    
    double vx, vy, vz;
    
    double x_over_dx0, x_over_dx1, y_over_dy0, y_over_dy1;
    
    double charge_density, factor, jx_buff;
    for (ipart = 0; ipart < npart; ipart++) {
        if (is_dead[ipart]) {
            continue;
        }
        vx = ux[ipart] * LIGHT_SPEED * inv_gamma[ipart];
        vy = uy[ipart] * LIGHT_SPEED * inv_gamma[ipart];
        vz = uz[ipart] * LIGHT_SPEED * inv_gamma[ipart];
        x_old = x[ipart] - vx * 0.5 * dt - x0;
        y_old = y[ipart] - vy * 0.5 * dt - y0;
        x_adv = x[ipart] + vx * 0.5 * dt - x0;
        y_adv = y[ipart] + vy * 0.5 * dt - y0;

        x_over_dx0 = x_old / dx;
        ix0 = (int)floor(x_over_dx0 + 0.5);
        y_over_dy0 = y_old / dy;
        iy0 = (int)floor(y_over_dy0 + 0.5);

        calculate_S(ix0 - x_over_dx0, 0, S0x);
        calculate_S(iy0 - y_over_dy0, 0, S0y);

        x_over_dx1 = x_adv / dx;
        ix1 = (int)floor(x_over_dx1 + 0.5);
        dcell_x = ix1 - ix0;

        y_over_dy1 = y_adv / dy;
        iy1 = (int)floor(y_over_dy1 + 0.5);
        dcell_y = iy1 - iy0;
        
        calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
        calculate_S(iy1 - y_over_dy1, dcell_y, S1y);

        for (i = 0; i < 5; i++) {
            DSx[i] = S1x[i] - S0x[i];
            DSy[i] = S1y[i] - S0y[i];
            jy_buff[i] = 0;
        }

        charge_density = q * w[ipart] / (dx * dy);
        factor = charge_density / dt;

        for (j = fmin(1, 1 + dcell_y); j < fmax(4, 4 + dcell_y); j++) {
            jx_buff = 0.0;
            iy = iy0 + (j - 2);
            if (iy < 0) {
                iy = ny + iy;
            }
            for (i = fmin(1, 1 + dcell_x); i < fmax(4, 4 + dcell_x); i++) {
                ix = ix0 + (i - 2);
                if (ix < 0) {
                    ix = nx + ix;
                }
                double wx = DSx[i] * (S0y[j] + 0.5 * DSy[j]);
                double wy = DSy[j] * (S0x[i] + 0.5 * DSx[i]);
                double wz = S0x[i] * S0y[j] + 0.5 * DSx[i] * S0y[j] + 0.5 * S0x[i] * DSy[j] + one_third * DSx[i] * DSy[j];

                jx_buff -= factor * dx * wx;
                jy_buff[i] -= factor * dy * wy;

                jx[INDEX(ix, iy, nx, ny)] += jx_buff;
                jy[INDEX(ix, iy, nx, ny)] += jy_buff[i];
                jz[INDEX(ix, iy, nx, ny)] += factor * dt * wz * vz;
                rho[INDEX(ix, iy, nx, ny)] += charge_density * S1x[i] * S1y[j];
            }
        }
    }
}

static PyObject* current_deposition_cpu(PyObject* self, PyObject* args) {
    PyObject *fields_list, *particles_list;
    npy_intp npatches;
    double dt, q;

    if (!PyArg_ParseTuple(args, "OOndd", 
        &fields_list, &particles_list,
        &npatches, &dt, &q)) {
        return NULL;
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;
    }
    
    npy_intp nx = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "nx"));
    npy_intp ny = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "ny"));
    double dx = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dx"));
    double dy = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dy"));

    double **rho        = (double**) malloc(npatches * sizeof(double*));
    double **jx         = (double**) malloc(npatches * sizeof(double*));
    double **jy         = (double**) malloc(npatches * sizeof(double*));
    double **jz         = (double**) malloc(npatches * sizeof(double*));
    double *x0          = (double*)  malloc(npatches * sizeof(double));
    double *y0          = (double*)  malloc(npatches * sizeof(double));

    double **x          = (double**) malloc(npatches * sizeof(double*));
    double **y          = (double**) malloc(npatches * sizeof(double*));
    double **ux         = (double**) malloc(npatches * sizeof(double*));
    double **uy         = (double**) malloc(npatches * sizeof(double*));
    double **uz         = (double**) malloc(npatches * sizeof(double*));
    double **inv_gamma  = (double**) malloc(npatches * sizeof(double*));
    npy_bool **is_dead  = (npy_bool**) malloc(npatches * sizeof(npy_bool*));
    double **w          = (double**) malloc(npatches * sizeof(double*));
    npy_intp *npart     = (npy_intp*) malloc(npatches * sizeof(npy_intp));

    // prestore the data in the list
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *fields = PyList_GetItem(fields_list, ipatch);
        PyObject *particles = PyList_GetItem(particles_list, ipatch);

        PyObject *rho_npy = PyObject_GetAttrString(fields, "rho");
        PyObject *jx_npy  = PyObject_GetAttrString(fields, "jx");
        PyObject *jy_npy  = PyObject_GetAttrString(fields, "jy");
        PyObject *jz_npy  = PyObject_GetAttrString(fields, "jz");
        x0[ipatch]        = PyFloat_AsDouble(PyObject_GetAttrString(fields, "x0"));
        y0[ipatch]        = PyFloat_AsDouble(PyObject_GetAttrString(fields, "y0"));

        PyObject *x_npy          = PyObject_GetAttrString(particles, "x");
        PyObject *y_npy          = PyObject_GetAttrString(particles, "y");
        PyObject *ux_npy         = PyObject_GetAttrString(particles, "ux");
        PyObject *uy_npy         = PyObject_GetAttrString(particles, "uy");
        PyObject *uz_npy         = PyObject_GetAttrString(particles, "uz");
        PyObject *inv_gamma_npy  = PyObject_GetAttrString(particles, "inv_gamma");
        PyObject *is_dead_npy    = PyObject_GetAttrString(particles, "is_dead");
        PyObject *w_npy          = PyObject_GetAttrString(particles, "w");

        rho[ipatch] = (double*) PyArray_DATA((PyArrayObject*) rho_npy);
        jx[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jx_npy);
        jy[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jy_npy);
        jz[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jz_npy);
        
        x[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) x_npy);
        y[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) y_npy);
        ux[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) ux_npy);
        uy[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) uy_npy);
        uz[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) uz_npy);
        inv_gamma[ipatch] = (double*) PyArray_DATA((PyArrayObject*) inv_gamma_npy);
        is_dead[ipatch]   = (npy_bool*) PyArray_DATA((PyArrayObject*) is_dead_npy);
        w[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) w_npy);

        npart[ipatch]     = PyArray_DIM((PyArrayObject*) w_npy, 0);

        Py_DecRef(rho_npy);
        Py_DecRef(jx_npy);
        Py_DecRef(jy_npy);
        Py_DecRef(jz_npy);
        Py_DecRef(x_npy);
        Py_DecRef(y_npy);
        Py_DecRef(ux_npy);
        Py_DecRef(uy_npy);
        Py_DecRef(uz_npy);
        Py_DecRef(inv_gamma_npy);
        Py_DecRef(is_dead_npy);
        Py_DecRef(w_npy);
    }

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        current_deposit_2d(
            rho[ipatch], jx[ipatch], jy[ipatch], jz[ipatch], 
            x[ipatch], y[ipatch], ux[ipatch], uy[ipatch], uz[ipatch], inv_gamma[ipatch], 
            is_dead[ipatch], 
            npart[ipatch], nx, ny,
            dx, dy, x0[ipatch], y0[ipatch], dt, w[ipatch], q
        );
    }
    Py_DecRef(fields_list);
    Py_DecRef(particles_list);
    free(rho);
    free(jx);
    free(jy);
    free(jz);
    free(x0);
    free(y0);
    free(x);
    free(y);
    free(ux);
    free(uy);
    free(uz);
    free(inv_gamma);
    free(is_dead);
    free(w);
    free(npart);
    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"current_deposition_cpu", current_deposition_cpu, METH_VARARGS, "Current deposition on CPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpumodule = {
    PyModuleDef_HEAD_INIT,
    "cpu",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_cpu(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
