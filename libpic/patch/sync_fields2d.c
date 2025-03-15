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
                0, 0, NY
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

static PyObject* sync_currents_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list;
    npy_intp npatches, nx, ny, nz, ng;

    if (!PyArg_ParseTuple(args, "OOnnnnn", 
        &fields_list, &patches_list,
        &npatches, &nx, &ny, &nz, &ng)) {
        return NULL;
    }
    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    npy_intp NZ = nz+2*ng;
    double **jx = get_attr_array_double(fields_list, npatches, "jx");
    double **jy = get_attr_array_double(fields_list, npatches, "jy"); 
    double **jz = get_attr_array_double(fields_list, npatches, "jz");
    double **rho = get_attr_array_double(fields_list, npatches, "rho");

    // faces
    npy_intp *xmin_index = get_attr_int(patches_list, npatches, "xmin_index");
    npy_intp *xmax_index = get_attr_int(patches_list, npatches, "xmax_index");
    npy_intp *ymin_index = get_attr_int(patches_list, npatches, "ymin_index");
    npy_intp *ymax_index = get_attr_int(patches_list, npatches, "ymax_index");
    npy_intp *zmin_index = get_attr_int(patches_list, npatches, "zmin_index");
    npy_intp *zmax_index = get_attr_int(patches_list, npatches, "zmax_index");
    // edges
    npy_intp *xminymin_index = get_attr_int(patches_list, npatches, "xminymin_index");
    npy_intp *xminymax_index = get_attr_int(patches_list, npatches, "xminymax_index");
    npy_intp *xminzmin_index = get_attr_int(patches_list, npatches, "xminymax_index");
    npy_intp *xminzmax_index = get_attr_int(patches_list, npatches, "xminymax_index");
    npy_intp *xmaxymin_index = get_attr_int(patches_list, npatches, "xmaxymin_index");
    npy_intp *xmaxymax_index = get_attr_int(patches_list, npatches, "xmaxymax_index");
    npy_intp *xmaxzmin_index = get_attr_int(patches_list, npatches, "xmaxzmin_index");
    npy_intp *xmaxzmax_index = get_attr_int(patches_list, npatches, "xmaxzmax_index");
    npy_intp *yminzmin_index = get_attr_int(patches_list, npatches, "yminzmin_index");
    npy_intp *yminzmax_index = get_attr_int(patches_list, npatches, "yminzmax_index");
    npy_intp *ymaxzmin_index = get_attr_int(patches_list, npatches, "ymaxzmin_index");
    npy_intp *ymaxzmax_index = get_attr_int(patches_list, npatches, "ymaxzmax_index");
    // corners
    npy_intp *xminyminzmin_index = get_attr_int(patches_list, npatches, "xminyminzmin_index");
    npy_intp *xminyminzmax_index = get_attr_int(patches_list, npatches, "xminyminzmax_index");
    npy_intp *xminymaxzmin_index = get_attr_int(patches_list, npatches, "xminymaxzmin_index");
    npy_intp *xminymaxzmax_index = get_attr_int(patches_list, npatches, "xminymaxzmax_index");
    npy_intp *xmaxyminzmin_index = get_attr_int(patches_list, npatches, "xmaxyminzmin_index");
    npy_intp *xmaxyminzmax_index = get_attr_int(patches_list, npatches, "xmaxyminzmax_index");
    npy_intp *xmaxymaxzmin_index = get_attr_int(patches_list, npatches, "xmaxymaxzmin_index");
    npy_intp *xmaxymaxzmax_index = get_attr_int(patches_list, npatches, "xmaxymaxzmax_index");


    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*4; i++) {
        int field_type = i % 4;
        npy_intp ipatch = i / 4;
        
        double** field = NULL;
        switch(field_type) {
            case 0: field = jx; break;
            case 1: field = jy; break;
            case 2: field = jz; break;
            case 3: field = rho; break;
        }

        #define SYNC_BOUNDARY_3D(src_patch, \
            ix_dst, ix_src, NX, \
            iy_dst, iy_src, NY, \
            iz_dst, iz_src, NZ) \
        for (npy_intp ix = 0; ix < NX; ix++) { \
            for (npy_intp iy = 0; iy < NY; iy++) { \
                for (npy_intp iz = 0; iz < NZ; iz++) { \
                    field[ipatch][INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] += field[src_patch][INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)]; \
                    field[src_patch][INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)] = 0.0; \
                } \
            } \
        }

        // X direction sync
        if (xmin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xmin_index[ipatch], 
                0, nx, ng, 
                0, 0, ny, 
                0, 0, nz
            )
        }
        
        if (xmax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xmax_index[ipatch], 
                nx-ng, -ng, ng, 
                0, 0, ny, 
                0, 0, nz
            )
        }

        // Y direction sync
        if (ymin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(ymin_index[ipatch], 
                0, 0, nx, 
                0, ny, ng, 
                0, 0, nz
            )
        }
        
        if (ymax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(ymax_index[ipatch], 
                0, 0, nx, 
                ny-ng, -ng, ng, 
                0, 0, nz
            )
        }

        // Z direction sync
        if (zmin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(zmin_index[ipatch], 
                0, 0, nx, 
                0, 0, ny, 
                0, nz, ng
            )
        }
        
        if (zmax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(zmax_index[ipatch], 
                0, 0, nx, 
                0, 0, ny, 
                nz-ng, -ng, ng
            )
        }

        // Edge synchronization (3D)
        if (xminymin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xminymin_index[ipatch], 
                0, nx, ng, 
                0, ny, ng, 
                0, 0, nz
            )
        }
        
        if (xminymax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xminymax_index[ipatch], 
                0, nx, ng, 
                ny-ng, -ng, ng, 
                0, 0, nz
            )
        }

        if (xminzmin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xminzmin_index[ipatch], 
                0, nx, ng, 
                0, 0, ny, 
                0, nz, ng
            )
        }

        if (xminzmax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xminzmax_index[ipatch], 
                0, nx, ng, 
                0, 0, ny, 
                nz-ng, -ng, ng
            )
        }

        if (xmaxymin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xmaxymin_index[ipatch],
                nx-ng, -ng, ng,
                0, ny, ng,
                0, 0, nz
            )
        }
        
        if (xmaxymax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xmaxymax_index[ipatch],
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                0, 0, nz
            )
        }

        if (xmaxzmin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xmaxzmin_index[ipatch],
                nx-ng, -ng, ng,
                0, 0, ny,
                0, nz, ng
            )
        }

        if (xmaxzmax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(xmaxzmax_index[ipatch],
                nx-ng, -ng, ng,
                0, 0, ny,
                nz-ng, -ng, ng
            )
        }

        if (yminzmin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(yminzmin_index[ipatch],
                0, 0, nx,
                0, ny, ng,
                0, nz, ng
            )
        }

        if (yminzmax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(yminzmax_index[ipatch],
                0, 0, nx,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        if (ymaxzmin_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(ymaxzmin_index[ipatch],
                0, 0, nx,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        if (ymaxzmax_index[ipatch] >= 0) {
            SYNC_BOUNDARY_3D(ymaxzmax_index[ipatch],
                0, 0, nx,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

        // Corner synchronization (3D)
        if (xminyminzmin_index[ipatch]) {
            SYNC_BOUNDARY_3D(xminyminzmin_index[ipatch],
                0, nx, ng,
                0, ny, ng,
                0, nz, ng
            )
        }

        if (xminyminzmax_index[ipatch]) {
            SYNC_BOUNDARY_3D(xminyminzmax_index[ipatch],
                0, nx, ng,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        if (xminymaxzmin_index[ipatch]) {
            SYNC_BOUNDARY_3D(xminymaxzmin_index[ipatch],
                0, nx, ng,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        if (xminymaxzmax_index[ipatch]) {
            SYNC_BOUNDARY_3D(xminymaxzmax_index[ipatch],
                0, nx, ng,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

        if (xmaxyminzmin_index[ipatch]) {
            SYNC_BOUNDARY_3D(xmaxyminzmin_index[ipatch],
                nx-ng, -ng, ng,
                0, ny, ng,
                0, nz, ng
            )
        }

        if (xmaxyminzmax_index[ipatch]) {
            SYNC_BOUNDARY_3D(xmaxyminzmax_index[ipatch],
                nx-ng, -ng, ng,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        if (xmaxymaxzmin_index[ipatch]) {
            SYNC_BOUNDARY_3D(xmaxymaxzmin_index[ipatch],
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        if (xmaxymaxzmax_index[ipatch]) {
            SYNC_BOUNDARY_3D(xmaxymaxzmax_index[ipatch],
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

    }
    Py_END_ALLOW_THREADS

    free(jx); free(jy); free(jz); free(rho);
    
    free(xmin_index); free(xmax_index); free(ymin_index); free(ymax_index); free(zmin_index); free(zmax_index);

    free(xminymin_index); free(xminymax_index); free(xminzmin_index); free(xminzmax_index); 
    free(xmaxymin_index); free(xmaxymax_index); free(xmaxzmin_index); free(xmaxzmax_index); 
    free(yminzmin_index); free(yminzmax_index); free(ymaxzmin_index); free(ymaxzmax_index);

    free(xminyminzmin_index); free(xminyminzmax_index); free(xminymaxzmin_index); free(xminymaxzmax_index);
    free(xmaxyminzmin_index); free(xmaxyminzmax_index); free(xmaxymaxzmin_index); free(xmaxymaxzmax_index);
    Py_DECREF(fields_list);
    Py_DECREF(patches_list);
    
    Py_RETURN_NONE;
}

static PyObject* sync_guard_fields_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list;
    PyArrayObject *xmin_index_arr, *xmax_index_arr, *ymin_index_arr, *ymax_index_arr, *zmin_index_arr, *zmax_index_arr;
    npy_intp npatches, nx, ny, nz, ng;

    if (!PyArg_ParseTuple(args, "OOOOOOnnnnn", 
        &fields_list, 
        &xmin_index_arr, &xmax_index_arr,
        &ymin_index_arr, &ymax_index_arr,
        &zmin_index_arr, &zmax_index_arr,
        &npatches, &nx, &ny, &nz, &ng)) {
        return NULL;
    }
    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    npy_intp NZ = nz+2*ng;
    double **ex = get_attr_array_double(fields_list, npatches, "ex");
    double **ey = get_attr_array_double(fields_list, npatches, "ey");
    double **ez = get_attr_array_double(fields_list, npatches, "ez");
    double **bx = get_attr_array_double(fields_list, npatches, "bx");
    double **by = get_attr_array_double(fields_list, npatches, "by");
    double **bz = get_attr_array_double(fields_list, npatches, "bz");

    const npy_intp* xmin_index = (npy_intp*)PyArray_DATA(xmin_index_arr);
    const npy_intp* xmax_index = (npy_intp*)PyArray_DATA(xmax_index_arr);
    const npy_intp* ymin_index = (npy_intp*)PyArray_DATA(ymin_index_arr);
    const npy_intp* ymax_index = (npy_intp*)PyArray_DATA(ymax_index_arr);
    const npy_intp* zmin_index = (npy_intp*)PyArray_DATA(zmin_index_arr);
    const npy_intp* zmax_index = (npy_intp*)PyArray_DATA(zmax_index_arr);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*6; i++) {
        int field_type = i % 6;
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

        npy_intp xmin_ipatch = xmin_index[ipatch];
        npy_intp xmax_ipatch = xmax_index[ipatch];
        npy_intp ymin_ipatch = ymin_index[ipatch];
        npy_intp ymax_ipatch = ymax_index[ipatch];
        npy_intp zmin_ipatch = zmin_index[ipatch];
        npy_intp zmax_ipatch = zmax_index[ipatch];

        // X direction sync
        if (xmin_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    for (npy_intp iz = 0; iz < nz; iz++) {
                        field[ipatch][INDEX3(-ixg-1, iy, iz)] = field[xmin_ipatch][INDEX3(nx-ng+ixg, iy, iz)];
                    }
                }
            }
        }
        
        if (xmax_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    for (npy_intp iz = 0; iz < nz; iz++) {
                        field[ipatch][INDEX3(nx+ixg, iy, iz)] = field[xmax_ipatch][INDEX3(ixg, iy, iz)];
                    }
                }
            }
        }

        // Y direction sync
        if (ymin_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    for (npy_intp iz = 0; iz < nz; iz++) {
                        field[ipatch][INDEX3(ix, -iyg-1, iz)] = field[ymin_ipatch][INDEX3(ix, ny-ng+iyg, iz)];
                    }
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    for (npy_intp iz = 0; iz < nz; iz++) {
                        field[ipatch][INDEX3(ix, ny+iyg, iz)] = field[ymax_ipatch][INDEX3(ix, iyg, iz)];
                    }
                }
            }
        }

        // Z direction sync
        if (zmin_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    for (npy_intp izg = 0; izg < ng; izg++) {
                        field[ipatch][INDEX3(ix, iy, -izg-1)] = field[zmin_ipatch][INDEX3(ix, iy, nz-ng+izg)];
                    }
                }
            }
        }
        
        if (zmax_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    for (npy_intp izg = 0; izg < ng; izg++) {
                        field[ipatch][INDEX3(ix, iy, nz+izg)] = field[zmax_ipatch][INDEX3(ix, iy, izg)];
                    }
                }
            }
        }

    }
    Py_END_ALLOW_THREADS

    free(ex); free(ey); free(ez);
    free(bx); free(by); free(bz);
    Py_DECREF(fields_list);
    
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"sync_currents_2d", sync_currents_2d, METH_VARARGS, "Synchronize currents between patches (2D)"},
    {"sync_guard_fields_2d", sync_guard_fields_2d, METH_VARARGS, "Synchronize guard cells between patches (2D)"},
    {"sync_currents_3d", sync_currents_3d, METH_VARARGS, "Synchronize currents between patches (3D)"},
    {"sync_guard_fields_3d", sync_guard_fields_3d, METH_VARARGS, "Synchronize guard cells between patches (3D)"},
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
