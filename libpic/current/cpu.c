#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define LIGHT_SPEED 299792458.0
#define one_third 0.3333333333333333

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
    
    int dcell_x, dcell_y, ix0, iy0, ix1, iy1, ix, iy;
    
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

                jx[iy + ny * ix] += jx_buff;
                jy[iy + ny * ix] += jy_buff[i];
                jz[iy + ny * ix] += factor * dt * wz * vz;
                rho[iy + ny * ix] += charge_density * S1x[i] * S1y[j];
            }
        }
    }
}

static PyObject* current_deposition_cpu(PyObject* self, PyObject* args) {
    PyObject *rho_list, *jx_list, *jy_list, *jz_list, *x0_list, *y0_list, *x_list, *y_list, *ux_list, *uy_list, *uz_list, *inv_gamma_list, *is_dead_list, *w_list;
    double dx, dy, dt, q;
    npy_intp npatches;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOndddd", 
        &rho_list, &jx_list, &jy_list, &jz_list, 
        &x0_list, &y0_list, 
        &x_list, &y_list, 
        &ux_list, &uy_list, &uz_list, 
        &inv_gamma_list, 
        &is_dead_list, 
        &w_list, 
        &npatches, &dx, &dy, &dt, &q)) {
        return NULL;
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;
    }

    double** rho_data = malloc(npatches * sizeof(double*));
    double** jx_data = malloc(npatches * sizeof(double*));
    double** jy_data = malloc(npatches * sizeof(double*));
    double** jz_data = malloc(npatches * sizeof(double*));
    double** x_data = malloc(npatches * sizeof(double*));
    double** y_data = malloc(npatches * sizeof(double*));
    double** ux_data = malloc(npatches * sizeof(double*));
    double** uy_data = malloc(npatches * sizeof(double*));
    double** uz_data = malloc(npatches * sizeof(double*));
    double** inv_gamma_data = malloc(npatches * sizeof(double*));
    npy_bool** is_dead_data = malloc(npatches * sizeof(npy_bool*));
    double** w_data = malloc(npatches * sizeof(double*));
    double* x0 = malloc(npatches * sizeof(double));
    double* y0 = malloc(npatches * sizeof(double));
    npy_intp* npart = malloc(npatches * sizeof(npy_intp));

    npy_intp nx = PyArray_DIM((PyArrayObject*)PyList_GetItem(jx_list, 0), 0);
    npy_intp ny = PyArray_DIM((PyArrayObject*)PyList_GetItem(jx_list, 0), 1);

    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyArrayObject *rho = (PyArrayObject*)PyList_GetItem(rho_list, ipatch);
        PyArrayObject *jx = (PyArrayObject*)PyList_GetItem(jx_list, ipatch);
        PyArrayObject *jy = (PyArrayObject*)PyList_GetItem(jy_list, ipatch);
        PyArrayObject *jz = (PyArrayObject*)PyList_GetItem(jz_list, ipatch);
        PyArrayObject *x = (PyArrayObject*)PyList_GetItem(x_list, ipatch);
        PyArrayObject *y = (PyArrayObject*)PyList_GetItem(y_list, ipatch);
        PyArrayObject *ux = (PyArrayObject*)PyList_GetItem(ux_list, ipatch);
        PyArrayObject *uy = (PyArrayObject*)PyList_GetItem(uy_list, ipatch);
        PyArrayObject *uz = (PyArrayObject*)PyList_GetItem(uz_list, ipatch);
        PyArrayObject *inv_gamma = (PyArrayObject*)PyList_GetItem(inv_gamma_list, ipatch);
        PyArrayObject *is_dead = (PyArrayObject*)PyList_GetItem(is_dead_list, ipatch);
        PyArrayObject *w = (PyArrayObject*)PyList_GetItem(w_list, ipatch);

        rho_data[ipatch] = (double*)PyArray_DATA(rho);
        jx_data[ipatch] = (double*)PyArray_DATA(jx);
        jy_data[ipatch] = (double*)PyArray_DATA(jy);
        jz_data[ipatch] = (double*)PyArray_DATA(jz);
        x_data[ipatch] = (double*)PyArray_DATA(x);
        y_data[ipatch] = (double*)PyArray_DATA(y);
        ux_data[ipatch] = (double*)PyArray_DATA(ux);
        uy_data[ipatch] = (double*)PyArray_DATA(uy);
        uz_data[ipatch] = (double*)PyArray_DATA(uz);
        inv_gamma_data[ipatch] = (double*)PyArray_DATA(inv_gamma);
        is_dead_data[ipatch] = (npy_bool*)PyArray_DATA(is_dead);
        w_data[ipatch] = (double*)PyArray_DATA(w);

        x0[ipatch] = PyFloat_AsDouble(PyList_GetItem(x0_list, ipatch));
        y0[ipatch] = PyFloat_AsDouble(PyList_GetItem(y0_list, ipatch));

        npart[ipatch] = PyArray_DIM(w, 0);
    }

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {

        current_deposit_2d(
            rho_data[ipatch], jx_data[ipatch], jy_data[ipatch], jz_data[ipatch], 
            x_data[ipatch], y_data[ipatch], ux_data[ipatch], uy_data[ipatch], uz_data[ipatch], inv_gamma_data[ipatch], 
            is_dead_data[ipatch], 
            npart[ipatch], nx, ny,
            dx, dy, x0[ipatch], y0[ipatch], dt, w_data[ipatch], q
        );
    }

    free(rho_data);
    free(jx_data);
    free(jy_data);
    free(jz_data);
    free(x_data);
    free(y_data);
    free(ux_data);
    free(uy_data);
    free(uz_data);
    free(inv_gamma_data);
    free(is_dead_data);
    free(w_data);
    free(x0);
    free(y0);
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
