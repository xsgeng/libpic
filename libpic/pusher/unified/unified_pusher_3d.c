#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#include "../../utils/cutils.h"


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

inline static void push_position_3d(
    double* x, double* y, double* z,
    double ux, double uy, double uz,
    double inv_gamma,
    double dt
) {
    const double cdt = LIGHT_SPEED * dt;
    *x += cdt * inv_gamma * ux;
    *y += cdt * inv_gamma * uy;
    *z += cdt * inv_gamma * uz;
}


inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

inline static double interp_field(
    double* field, 
    double* facx, double* facy, double* facz, 
    npy_intp ix, npy_intp iy, npy_intp iz, 
    npy_intp nx, npy_intp ny, npy_intp nz
) {
    double field_part = 
          facz[0] * (facy[0] * (facx[0] * field[INDEX3(ix-1, iy-1, iz-1)]
        +                       facx[1] * field[INDEX3(ix  , iy-1, iz-1)]
        +                       facx[2] * field[INDEX3(ix+1, iy-1, iz-1)])
        +            facy[1] * (facx[0] * field[INDEX3(ix-1, iy  , iz-1)]
        +                       facx[1] * field[INDEX3(ix  , iy  , iz-1)]
        +                       facx[2] * field[INDEX3(ix+1, iy  , iz-1)])
        +            facy[2] * (facx[0] * field[INDEX3(ix-1, iy+1, iz-1)]
        +                       facx[1] * field[INDEX3(ix  , iy+1, iz-1)]
        +                       facx[2] * field[INDEX3(ix+1, iy+1, iz-1)]))
        + facz[1] * (facy[0] * (facx[0] * field[INDEX3(ix-1, iy-1, iz  )]
        +                       facx[1] * field[INDEX3(ix  , iy-1, iz  )]
        +                       facx[2] * field[INDEX3(ix+1, iy-1, iz  )])
        +            facy[1] * (facx[0] * field[INDEX3(ix-1, iy  , iz  )]
        +                       facx[1] * field[INDEX3(ix  , iy  , iz  )]
        +                       facx[2] * field[INDEX3(ix+1, iy  , iz  )])
        +            facy[2] * (facx[0] * field[INDEX3(ix-1, iy+1, iz  )]
        +                       facx[1] * field[INDEX3(ix  , iy+1, iz  )]
        +                       facx[2] * field[INDEX3(ix+1, iy+1, iz  )]))
        + facz[2] * (facy[0] * (facx[0] * field[INDEX3(ix-1, iy-1, iz+1)]
        +                       facx[1] * field[INDEX3(ix  , iy-1, iz+1)]
        +                       facx[2] * field[INDEX3(ix+1, iy-1, iz+1)])
        +            facy[1] * (facx[0] * field[INDEX3(ix-1, iy  , iz+1)]
        +                       facx[1] * field[INDEX3(ix  , iy  , iz+1)]
        +                       facx[2] * field[INDEX3(ix+1, iy  , iz+1)])
        +            facy[2] * (facx[0] * field[INDEX3(ix-1, iy+1, iz+1)]
        +                       facx[1] * field[INDEX3(ix  , iy+1, iz+1)]
        +                       facx[2] * field[INDEX3(ix+1, iy+1, iz+1)]));
    return field_part;
}

inline static void interpolation_3d(
    double x, double y, double z,
    double* ex_part, double* ey_part, double* ez_part, 
    double* bx_part, double* by_part, double* bz_part, 
    double* ex, double* ey, double* ez, 
    double* bx, double* by, double* bz, 
    double dx, double dy, double dz, double x0, double y0, double z0, 
    npy_intp nx, npy_intp ny, npy_intp nz
) {
    
    double gx[3];
    double gy[3];
    double gz[3];
    double hx[3];
    double hy[3];
    double hz[3];
    
    double x_over_dx = (x - x0) / dx;
    double y_over_dy = (y - y0) / dy;
    double z_over_dz = (z - z0) / dz;

    npy_intp ix1 = (int)floor(x_over_dx + 0.5);
    get_gx(ix1 - x_over_dx, gx);

    npy_intp ix2 = (int)floor(x_over_dx);
    get_gx(ix2 - x_over_dx + 0.5, hx);

    npy_intp iy1 = (int)floor(y_over_dy + 0.5);
    get_gx(iy1 - y_over_dy, gy);

    npy_intp iy2 = (int)floor(y_over_dy);
    get_gx(iy2 - y_over_dy + 0.5, hy);

    npy_intp iz1 = (int)floor(z_over_dz + 0.5);
    get_gx(iz1 - z_over_dz, gz);

    npy_intp iz2 = (int)floor(z_over_dz);
    get_gx(iz2 - z_over_dz + 0.5, hz);

    *ex_part = interp_field(ex, hx, gy, gz, ix2, iy1, iz1, nx, ny, nz);
    *ey_part = interp_field(ey, gx, hy, gz, ix1, iy2, iz1, nx, ny, nz);
    *ez_part = interp_field(ez, gx, gy, hz, ix1, iy1, iz2, nx, ny, nz);

    *bx_part = interp_field(bx, gx, hy, hz, ix1, iy2, iz2, nx, ny, nz);
    *by_part = interp_field(by, hx, gy, hz, ix2, iy1, iz2, nx, ny, nz);
    *bz_part = interp_field(bz, hx, hy, gz, ix2, iy2, iz1, nx, ny, nz);
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
inline static void current_deposit_3d(
    double* rho, double* jx, double* jy, double* jz, 
    double x, double y, double z,
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny, npy_intp nz,
    double dx, double dy, double dz, double x0, double y0, double z0, double dt, double w, double q
) {
    double vx = ux * LIGHT_SPEED * inv_gamma;
    double vy = uy * LIGHT_SPEED * inv_gamma;
    double vz = uz * LIGHT_SPEED * inv_gamma;
    double x_old = x - vx * 0.5 * dt - x0;
    double x_adv = x + vx * 0.5 * dt - x0;
    double y_old = y - vy * 0.5 * dt - y0;
    double y_adv = y + vy * 0.5 * dt - y0;
    double z_old = z - vz * 0.5 * dt - z0;
    double z_adv = z + vz * 0.5 * dt - z0;

    double x_over_dx0 = x_old / dx;
    int ix0 = (int)floor(x_over_dx0 + 0.5);
    double y_over_dy0 = y_old / dy;
    int iy0 = (int)floor(y_over_dy0 + 0.5);
    double z_over_dz0 = z_old / dz;
    int iz0 = (int)floor(z_over_dz0 + 0.5);

    double S0x[5], S0y[5], S0z[5];
    calculate_S(ix0 - x_over_dx0, 0, S0x);
    calculate_S(iy0 - y_over_dy0, 0, S0y);
    calculate_S(iz0 - z_over_dz0, 0, S0z);

    double x_over_dx1 = x_adv / dx;
    int ix1 = (int)floor(x_over_dx1 + 0.5);
    int dcell_x = ix1 - ix0;

    double y_over_dy1 = y_adv / dy;
    int iy1 = (int)floor(y_over_dy1 + 0.5);
    int dcell_y = iy1 - iy0;

    double z_over_dz1 = z_adv / dz;
    int iz1 = (int)floor(z_over_dz1 + 0.5);
    int dcell_z = iz1 - iz0;
    
    double S1x[5], S1y[5], S1z[5], DSx[5], DSy[5], DSz[5];
    calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    calculate_S(iy1 - y_over_dy1, dcell_y, S1y);
    calculate_S(iz1 - z_over_dz1, dcell_z, S1z);

    for (int i = 0; i < 5; i++) {
        DSx[i] = S1x[i] - S0x[i];
        DSy[i] = S1y[i] - S0y[i];
        DSz[i] = S1z[i] - S0z[i];
    }

    double charge_density = q * w / (dx * dy);
    double factor = charge_density / dt;

    double jx_buff[5][5] = {0};
    for (int i = fmin(1, 1 + dcell_x); i < fmax(4, 4 + dcell_x); i++) {
        int ix = ix0 + (i - 2);
        
        double jy_buff[5] = {0};
        for (int j = fmin(1, 1 + dcell_y); j < fmax(4, 4 + dcell_y); j++) {
            int iy = iy0 + (j - 2);
            
            double jz_buff = 0;
            for (int k = fmin(1, 1 + dcell_z); k < fmax(4, 4 + dcell_z); k++) {
                int iz = iz0 + (k - 2);

                double wx = DSx[i] * (S0y[j]*S0z[k] + 0.5 * DSy[j]*S0z[k] + 0.5*S0y[j]*DSz[k] + one_third*DSy[j]*DSz[k]);
                double wy = DSy[j] * (S0x[i]*S0z[k] + 0.5 * DSx[i]*S0z[k] + 0.5*S0x[i]*DSz[k] + one_third*DSx[i]*DSz[k]);
                double wz = DSz[k] * (S0x[i]*S0y[j] + 0.5 * DSx[i]*S0y[j] + 0.5*S0x[i]*DSy[j] + one_third*DSx[i]*DSy[j]);

                jx_buff[k][j] -= factor * dx * wx;
                jy_buff[k] -= factor * dy * wy;
                jz_buff -= factor * dz * wz;

                jx[INDEX3(ix, iy, iz)] += jx_buff[k][j];
                jy[INDEX3(ix, iy, iz)] += jy_buff[k];
                jz[INDEX3(ix, iy, iz)] += jz_buff;
                rho[INDEX3(ix, iy, iz)] += charge_density * S1x[i] * S1y[j] * S1z[k];
            }
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
    npy_intp nz = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "nz"));
    npy_intp n_guard = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "n_guard"));
    nx += 2*n_guard;
    ny += 2*n_guard;
    nz += 2*n_guard;
    double dx = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dx"));
    double dy = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dy"));
    double dz = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dz"));

    // fields
    double **ex         = get_attr_array_double(fields_list, npatches, "ex");
    double **ey         = get_attr_array_double(fields_list, npatches, "ey");
    double **ez         = get_attr_array_double(fields_list, npatches, "ez");
    double **bx         = get_attr_array_double(fields_list, npatches, "bx");
    double **by         = get_attr_array_double(fields_list, npatches, "by");
    double **bz         = get_attr_array_double(fields_list, npatches, "bz");
    double **rho        = get_attr_array_double(fields_list, npatches, "rho");
    double **jx         = get_attr_array_double(fields_list, npatches, "jx");
    double **jy         = get_attr_array_double(fields_list, npatches, "jy");
    double **jz         = get_attr_array_double(fields_list, npatches, "jz");
    double *x0          = get_attr_double(fields_list, npatches, "x0");
    double *y0          = get_attr_double(fields_list, npatches, "y0");
    double *z0          = get_attr_double(fields_list, npatches, "z0");

    // particles
    double **x          = get_attr_array_double(particles_list, npatches, "x");
    double **y          = get_attr_array_double(particles_list, npatches, "y");
    double **z          = get_attr_array_double(particles_list, npatches, "z");
    double **ux         = get_attr_array_double(particles_list, npatches, "ux");
    double **uy         = get_attr_array_double(particles_list, npatches, "uy");
    double **uz         = get_attr_array_double(particles_list, npatches, "uz");
    double **inv_gamma  = get_attr_array_double(particles_list, npatches, "inv_gamma");
    double **ex_part    = get_attr_array_double(particles_list, npatches, "ex_part");
    double **ey_part    = get_attr_array_double(particles_list, npatches, "ey_part");
    double **ez_part    = get_attr_array_double(particles_list, npatches, "ez_part");
    double **bx_part    = get_attr_array_double(particles_list, npatches, "bx_part");
    double **by_part    = get_attr_array_double(particles_list, npatches, "by_part");
    double **bz_part    = get_attr_array_double(particles_list, npatches, "bz_part");
    npy_bool **is_dead  = get_attr_array_bool(particles_list, npatches, "is_dead");
    double **w          = get_attr_array_double(particles_list, npatches, "w");
    npy_intp *npart     = get_attr_int(particles_list, npatches, "npart");

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ip = 0; ip < npart[ipatch]; ip++) {
            if (is_dead[ipatch][ip]) continue;
            if (isnan(x[ipatch][ip]) || isnan(y[ipatch][ip]) || isnan(z[ipatch][ip])) continue;
            
            push_position_3d(
                &x[ipatch][ip], &y[ipatch][ip], &z[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], uz[ipatch][ip], inv_gamma[ipatch][ip], 
                0.5*dt
            );
            interpolation_3d(
                x[ipatch][ip], y[ipatch][ip], z[ipatch][ip], 
                &ex_part[ipatch][ip], &ey_part[ipatch][ip], &ez_part[ipatch][ip], 
                &bx_part[ipatch][ip], &by_part[ipatch][ip], &bz_part[ipatch][ip], 
                ex[ipatch], ey[ipatch], ez[ipatch], 
                bx[ipatch], by[ipatch], bz[ipatch], 
                dx, dy, dz, x0[ipatch], y0[ipatch], z0[ipatch],
                nx, ny, nz
            );
            boris(
                &ux[ipatch][ip], &uy[ipatch][ip], &uz[ipatch][ip], &inv_gamma[ipatch][ip],
                ex_part[ipatch][ip], ey_part[ipatch][ip], ez_part[ipatch][ip], 
                bx_part[ipatch][ip], by_part[ipatch][ip], bz_part[ipatch][ip],
                q, m, dt
            );
            push_position_3d(
                &x[ipatch][ip], &y[ipatch][ip], &z[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], uz[ipatch][ip], inv_gamma[ipatch][ip], 
                0.5*dt
            );
            current_deposit_3d(
                rho[ipatch], jx[ipatch], jy[ipatch], jz[ipatch], 
                x[ipatch][ip], y[ipatch][ip], z[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], uz[ipatch][ip], inv_gamma[ipatch][ip],
                nx, ny, nz,
                dx, dy, dz, x0[ipatch], y0[ipatch], z0[ipatch], dt, w[ipatch][ip], q
            );
        }

    }
    // acquire GIL
    Py_END_ALLOW_THREADS

    Py_DecRef(fields_list);
    Py_DecRef(particles_list);
    // fields
    free(ex); free(ey); free(ez);
    free(bx); free(by); free(bz);
    free(rho); free(jx); free(jy); free(jz);
    free(x0); free(y0); free(z0);
    // particles
    free(x); free(y); free(z); free(ux); free(uy); free(uz); free(inv_gamma);
    free(ex_part); free(ey_part); free(ez_part);
    free(bx_part); free(by_part); free(bz_part);
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
    "unified_pusher_3d",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_cpu(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
