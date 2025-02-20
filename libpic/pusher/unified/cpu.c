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
#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))

inline static void boris(
    double* ux, double* uy, double* uz, double* inv_gamma,
    double Ex, double Ey, double Ez,
    double Bx, double By, double Bz,
    double q, double m, double dt
) {
    const double efactor = q * dt / (2 * m * LIGHT_SPEED);
    const double bfactor = q * dt / (2 * m);

    // E field half acceleration
    double ux_minus = *ux + efactor * Ex;
    double uy_minus = *uy + efactor * Ey;
    double uz_minus = *uz + efactor * Ez;

    // B field rotation
    *inv_gamma = 1.0 / sqrt(1 + ux_minus * ux_minus + uy_minus * uy_minus + uz_minus * uz_minus);
    double Tx = bfactor * Bx * (*inv_gamma);
    double Ty = bfactor * By * (*inv_gamma);
    double Tz = bfactor * Bz * (*inv_gamma);

    double ux_prime = ux_minus + uy_minus * Tz - uz_minus * Ty;
    double uy_prime = uy_minus + uz_minus * Tx - ux_minus * Tz;
    double uz_prime = uz_minus + ux_minus * Ty - uy_minus * Tx;

    double Tfactor = 2.0 / (1 + Tx * Tx + Ty * Ty + Tz * Tz);
    double Sx = Tfactor * Tx;
    double Sy = Tfactor * Ty;
    double Sz = Tfactor * Tz;

    double ux_plus = ux_minus + uy_prime * Sz - uz_prime * Sy;
    double uy_plus = uy_minus + uz_prime * Sx - ux_prime * Sz;
    double uz_plus = uz_minus + ux_prime * Sy - uy_prime * Sx;

    // E field half acceleration
    *ux = ux_plus + efactor * Ex;
    *uy = uy_plus + efactor * Ey;
    *uz = uz_plus + efactor * Ez;
    *inv_gamma = 1.0 / sqrt(1 + (*ux) * (*ux) + (*uy) * (*uy) + (*uz) * (*uz));
}

inline static void push_position_2d(
    double* x, double* y,
    double ux, double uy,
    double inv_gamma,
    double dt
) {
    const double cdt = LIGHT_SPEED * dt;
    *x += cdt * inv_gamma * ux;
    *y += cdt * inv_gamma * uy;
}


inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

inline static double interp_field(double* field, double* fac1, double* fac2, npy_intp ix, npy_intp iy, npy_intp nx, npy_intp ny) {
    double field_part = 
          fac2[0] * (fac1[0] * field[INDEX(ix-1, iy-1)] 
        +            fac1[1] * field[INDEX(ix,   iy-1)] 
        +            fac1[2] * field[INDEX(ix+1, iy-1)])
        + fac2[1] * (fac1[0] * field[INDEX(ix-1, iy  )] 
        +            fac1[1] * field[INDEX(ix,   iy  )] 
        +            fac1[2] * field[INDEX(ix+1, iy  )]) 
        + fac2[2] * (fac1[0] * field[INDEX(ix-1, iy+1)] 
        +            fac1[1] * field[INDEX(ix,   iy+1)] 
        +            fac1[2] * field[INDEX(ix+1, iy+1)]);
    return field_part;
}

inline static void interpolation_2d(
    double x, double y, 
    double* ex_part, double* ey_part, double* ez_part, 
    double* bx_part, double* by_part, double* bz_part, 
    double* ex, double* ey, double* ez, 
    double* bx, double* by, double* bz, 
    double dx, double dy, double x0, double y0, 
    npy_intp nx, npy_intp ny
) {
    
    double gx[3];
    double gy[3];
    double hx[3];
    double hy[3];
    
    double x_over_dx = (x - x0) / dx;
    double y_over_dy = (y - y0) / dy;

    npy_intp ix1 = (int)floor(x_over_dx + 0.5);
    get_gx(ix1 - x_over_dx, gx);

    npy_intp ix2 = (int)floor(x_over_dx);
    get_gx(ix2 - x_over_dx + 0.5, hx);

    npy_intp iy1 = (int)floor(y_over_dy + 0.5);
    get_gx(iy1 - y_over_dy, gy);

    npy_intp iy2 = (int)floor(y_over_dy);
    get_gx(iy2 - y_over_dy + 0.5, hy);

    *ex_part = interp_field(ex, hx, gy, ix2, iy1, nx, ny);
    *ey_part = interp_field(ey, gx, hy, ix1, iy2, nx, ny);
    *ez_part = interp_field(ez, gx, gy, ix1, iy1, nx, ny);

    *bx_part = interp_field(bx, gx, hy, ix1, iy2, nx, ny);
    *by_part = interp_field(by, hx, gy, ix2, iy1, nx, ny);
    *bz_part = interp_field(bz, hx, hy, ix2, iy2, nx, ny);
}

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
inline static void current_deposit_2d(
    double* rho, double* jx, double* jy, double* jz, 
    double x, double y, 
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny,
    double dx, double dy, double x0, double y0, double dt, double w, double q
) {
    double vx, vy, vz;
    double x_old, y_old, x_adv, y_adv;
    double S0x[5], S1x[5], S0y[5], S1y[5], DSx[5], DSy[5], jy_buff[5];
    
    npy_intp i, j;
    
    npy_intp dcell_x, dcell_y, ix0, iy0, ix1, iy1, ix, iy;
    
    double x_over_dx0, x_over_dx1, y_over_dy0, y_over_dy1;
    
    double charge_density, factor, jx_buff;

    vx = ux * LIGHT_SPEED * inv_gamma;
    vy = uy * LIGHT_SPEED * inv_gamma;
    vz = uz * LIGHT_SPEED * inv_gamma;
    x_old = x - vx * 0.5 * dt - x0;
    y_old = y - vy * 0.5 * dt - y0;
    x_adv = x + vx * 0.5 * dt - x0;
    y_adv = y + vy * 0.5 * dt - y0;

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

    charge_density = q * w / (dx * dy);
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

            jx[INDEX(ix, iy)] += jx_buff;
            jy[INDEX(ix, iy)] += jy_buff[i];
            jz[INDEX(ix, iy)] += factor * dt * wz * vz;
            rho[INDEX(ix, iy)] += charge_density * S1x[i] * S1y[j];
        }
    }
}

static PyObject* unified_boris_pusher_cpu(PyObject* self, PyObject* args) {
    PyObject *fields_list, *particles_list;
    npy_intp npatches;
    double dt, q, m;

    if (!PyArg_ParseTuple(args, "OOnddd", 
        &particles_list, &fields_list, 
        &npatches, &dt, &q, &m)) {
        return NULL;
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;
    }
    
    npy_intp nx = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "nx"));
    npy_intp ny = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "ny"));
    npy_intp n_guard = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "n_guard"));
    nx += 2*n_guard;
    ny += 2*n_guard;
    double dx = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dx"));
    double dy = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dy"));

    // fields
    double **ex         = (double**) malloc(npatches * sizeof(double*));
    double **ey         = (double**) malloc(npatches * sizeof(double*));
    double **ez         = (double**) malloc(npatches * sizeof(double*));
    double **bx         = (double**) malloc(npatches * sizeof(double*));
    double **by         = (double**) malloc(npatches * sizeof(double*));
    double **bz         = (double**) malloc(npatches * sizeof(double*));
    double **rho        = (double**) malloc(npatches * sizeof(double*));
    double **jx         = (double**) malloc(npatches * sizeof(double*));
    double **jy         = (double**) malloc(npatches * sizeof(double*));
    double **jz         = (double**) malloc(npatches * sizeof(double*));
    double *x0          = (double*)  malloc(npatches * sizeof(double));
    double *y0          = (double*)  malloc(npatches * sizeof(double));

    // particles
    double **x          = (double**) malloc(npatches * sizeof(double*));
    double **y          = (double**) malloc(npatches * sizeof(double*));
    double **ux         = (double**) malloc(npatches * sizeof(double*));
    double **uy         = (double**) malloc(npatches * sizeof(double*));
    double **uz         = (double**) malloc(npatches * sizeof(double*));
    double **inv_gamma  = (double**) malloc(npatches * sizeof(double*));
    double **ex_part    = (double**) malloc(npatches * sizeof(double*));
    double **ey_part    = (double**) malloc(npatches * sizeof(double*));
    double **ez_part    = (double**) malloc(npatches * sizeof(double*));
    double **bx_part    = (double**) malloc(npatches * sizeof(double*));
    double **by_part    = (double**) malloc(npatches * sizeof(double*));
    double **bz_part    = (double**) malloc(npatches * sizeof(double*));
    npy_bool **is_dead  = (npy_bool**) malloc(npatches * sizeof(npy_bool*));
    double **w          = (double**) malloc(npatches * sizeof(double*));
    npy_intp *npart     = (npy_intp*) malloc(npatches * sizeof(npy_intp));

    // prestore the data in the list
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *fields = PyList_GetItem(fields_list, ipatch);
        PyObject *particles = PyList_GetItem(particles_list, ipatch);

        // fields
        PyObject *ex_npy = PyObject_GetAttrString(fields, "ex");
        PyObject *ey_npy = PyObject_GetAttrString(fields, "ey");
        PyObject *ez_npy = PyObject_GetAttrString(fields, "ez");
        PyObject *bx_npy = PyObject_GetAttrString(fields, "bx");
        PyObject *by_npy = PyObject_GetAttrString(fields, "by");
        PyObject *bz_npy = PyObject_GetAttrString(fields, "bz");
        PyObject *rho_npy = PyObject_GetAttrString(fields, "rho");
        PyObject *jx_npy  = PyObject_GetAttrString(fields, "jx");
        PyObject *jy_npy  = PyObject_GetAttrString(fields, "jy");
        PyObject *jz_npy  = PyObject_GetAttrString(fields, "jz");
        x0[ipatch]        = PyFloat_AsDouble(PyObject_GetAttrString(fields, "x0"));
        y0[ipatch]        = PyFloat_AsDouble(PyObject_GetAttrString(fields, "y0"));

        ex[ipatch]     = (double*) PyArray_DATA((PyArrayObject*) ex_npy);
        ey[ipatch]     = (double*) PyArray_DATA((PyArrayObject*) ey_npy);
        ez[ipatch]     = (double*) PyArray_DATA((PyArrayObject*) ez_npy);
        bx[ipatch]     = (double*) PyArray_DATA((PyArrayObject*) bx_npy);
        by[ipatch]     = (double*) PyArray_DATA((PyArrayObject*) by_npy);
        bz[ipatch]     = (double*) PyArray_DATA((PyArrayObject*) bz_npy);
        rho[ipatch] = (double*) PyArray_DATA((PyArrayObject*) rho_npy);
        jx[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jx_npy);
        jy[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jy_npy);
        jz[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jz_npy);

        Py_DecRef(ex_npy);
        Py_DecRef(ey_npy);
        Py_DecRef(ez_npy);
        Py_DecRef(bx_npy);
        Py_DecRef(by_npy);
        Py_DecRef(bz_npy);
        Py_DecRef(rho_npy);
        Py_DecRef(jx_npy);
        Py_DecRef(jy_npy);
        Py_DecRef(jz_npy);

        // particles
        PyObject *x_npy          = PyObject_GetAttrString(particles, "x");
        PyObject *y_npy          = PyObject_GetAttrString(particles, "y");
        PyObject *ux_npy         = PyObject_GetAttrString(particles, "ux");
        PyObject *uy_npy         = PyObject_GetAttrString(particles, "uy");
        PyObject *uz_npy         = PyObject_GetAttrString(particles, "uz");
        PyObject *inv_gamma_npy  = PyObject_GetAttrString(particles, "inv_gamma");
        PyObject *ex_part_npy    = PyObject_GetAttrString(particles, "ex_part");
        PyObject *ey_part_npy    = PyObject_GetAttrString(particles, "ey_part");
        PyObject *ez_part_npy    = PyObject_GetAttrString(particles, "ez_part");
        PyObject *bx_part_npy    = PyObject_GetAttrString(particles, "bx_part");
        PyObject *by_part_npy    = PyObject_GetAttrString(particles, "by_part");
        PyObject *bz_part_npy    = PyObject_GetAttrString(particles, "bz_part");
        PyObject *is_dead_npy    = PyObject_GetAttrString(particles, "is_dead");
        PyObject *w_npy          = PyObject_GetAttrString(particles, "w");

        x[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) x_npy);
        y[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) y_npy);
        ux[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) ux_npy);
        uy[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) uy_npy);
        uz[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) uz_npy);
        inv_gamma[ipatch] = (double*) PyArray_DATA((PyArrayObject*) inv_gamma_npy);
        ex_part[ipatch]   = (double*) PyArray_DATA((PyArrayObject*) ex_part_npy);
        ey_part[ipatch]   = (double*) PyArray_DATA((PyArrayObject*) ey_part_npy);
        ez_part[ipatch]   = (double*) PyArray_DATA((PyArrayObject*) ez_part_npy);
        bx_part[ipatch]   = (double*) PyArray_DATA((PyArrayObject*) bx_part_npy);
        by_part[ipatch]   = (double*) PyArray_DATA((PyArrayObject*) by_part_npy);
        bz_part[ipatch]   = (double*) PyArray_DATA((PyArrayObject*) bz_part_npy);
        is_dead[ipatch]   = (npy_bool*) PyArray_DATA((PyArrayObject*) is_dead_npy);
        w[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) w_npy);

        npart[ipatch]     = PyArray_DIM((PyArrayObject*) w_npy, 0);

        Py_DecRef(x_npy);
        Py_DecRef(y_npy);
        Py_DecRef(ux_npy);
        Py_DecRef(uy_npy);
        Py_DecRef(uz_npy);
        Py_DecRef(inv_gamma_npy);
        Py_DecRef(ex_part_npy);
        Py_DecRef(ey_part_npy);
        Py_DecRef(ez_part_npy);
        Py_DecRef(bx_part_npy);
        Py_DecRef(by_part_npy);
        Py_DecRef(bz_part_npy);
        Py_DecRef(is_dead_npy);
        Py_DecRef(w_npy);
    }

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ip = 0; ip < npart[ipatch]; ip++) {
            if (is_dead[ipatch][ip]) continue;
            if (isnan(x[ipatch][ip]) || isnan(y[ipatch][ip])) continue;
            
            push_position_2d(
                &x[ipatch][ip], &y[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], inv_gamma[ipatch][ip], 
                0.5*dt
            );
            interpolation_2d(
                x[ipatch][ip], y[ipatch][ip], 
                &ex_part[ipatch][ip], &ey_part[ipatch][ip], &ez_part[ipatch][ip], 
                &bx_part[ipatch][ip], &by_part[ipatch][ip], &bz_part[ipatch][ip], 
                ex[ipatch], ey[ipatch], ez[ipatch], 
                bx[ipatch], by[ipatch], bz[ipatch], 
                dx, dy, x0[ipatch], y0[ipatch], 
                nx, ny
            );
            boris(
                &ux[ipatch][ip], &uy[ipatch][ip], &uz[ipatch][ip], &inv_gamma[ipatch][ip],
                ex_part[ipatch][ip], ey_part[ipatch][ip], ez_part[ipatch][ip], 
                bx_part[ipatch][ip], by_part[ipatch][ip], bz_part[ipatch][ip],
                q, m, dt
            );
            push_position_2d(
                &x[ipatch][ip], &y[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], inv_gamma[ipatch][ip], 
                0.5*dt
            );
            current_deposit_2d(
                rho[ipatch], jx[ipatch], jy[ipatch], jz[ipatch], 
                x[ipatch][ip], y[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], uz[ipatch][ip], inv_gamma[ipatch][ip],
                nx, ny,
                dx, dy, x0[ipatch], y0[ipatch], dt, w[ipatch][ip], q
            );
        }

    }
    // acquire GIL
    Py_END_ALLOW_THREADS

    Py_DecRef(fields_list);
    Py_DecRef(particles_list);
    // fields
    free(ex);
    free(ey);
    free(ez);
    free(bx);
    free(by);
    free(bz);
    free(rho);
    free(jx);
    free(jy);
    free(jz);
    free(x0);
    free(y0);
    // particles
    free(x);
    free(y);
    free(ux);
    free(uy);
    free(uz);
    free(inv_gamma);
    free(ex_part);
    free(ey_part);
    free(ez_part);
    free(bx_part);
    free(by_part);
    free(bz_part);
    free(is_dead);
    free(w);
    free(npart);
    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"unified_boris_pusher_cpu", unified_boris_pusher_cpu, METH_VARARGS, "Unified Boris Pusher"},
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
