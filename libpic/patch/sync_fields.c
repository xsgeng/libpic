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
    const npy_intp* xmin_index = (npy_intp*)PyArray_DATA(xmin_index_arr);
    const npy_intp* xmax_index = (npy_intp*)PyArray_DATA(xmax_index_arr);
    const npy_intp* ymin_index = (npy_intp*)PyArray_DATA(ymin_index_arr);
    const npy_intp* ymax_index = (npy_intp*)PyArray_DATA(ymax_index_arr);

    // Release GIL
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

        npy_intp xmin_ipatch = xmin_index[ipatch];
        npy_intp xmax_ipatch = xmax_index[ipatch];
        npy_intp ymin_ipatch = ymin_index[ipatch];
        npy_intp ymax_ipatch = ymax_index[ipatch];

        // X direction sync
        if (xmin_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    field[ipatch][INDEX2(ixg, iy)] += field[xmin_ipatch][INDEX2(nx+ixg, iy)];
                    field[xmin_ipatch][INDEX2(nx+ixg, iy)] = 0.0;
                }
            }
        }
        
        if (xmax_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    field[ipatch][INDEX2(nx-ng+ixg, iy)] += field[xmax_ipatch][INDEX2(-ng+ixg, iy)];
                    field[xmax_ipatch][INDEX2(-ng+ixg, iy)] = 0.0;
                }
            }
        }

        // Y direction sync
        if (ymin_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    field[ipatch][INDEX2(ix, iyg)] += field[ymin_ipatch][INDEX2(ix, ny+iyg)];
                    field[ymin_ipatch][INDEX2(ix, ny+iyg)] = 0.0;
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    field[ipatch][INDEX2(ix, ny-ng+iyg)] += field[ymax_ipatch][INDEX2(ix, -ng+iyg)];
                    field[ymax_ipatch][INDEX2(ix, -ng+iyg)] = 0.0;
                }
            }
        }

        // Corner synchronization
        if (ymin_ipatch >= 0) {
            npy_intp xminymin_ipatch = xmin_index[ymin_ipatch];
            if (xminymin_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(ixg, iyg)] += field[xminymin_ipatch][INDEX2(nx+ixg, ny+iyg)];
                        field[xminymin_ipatch][INDEX2(nx+ixg, ny+iyg)] = 0.0;
                    }
                }
            }
            
            npy_intp xmaxymin_ipatch = xmax_index[ymin_ipatch];
            if (xmaxymin_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(nx-ng+ixg, iyg)] += field[xmaxymin_ipatch][INDEX2(-ng+ixg, ny+iyg)];
                        field[xmaxymin_ipatch][INDEX2(-ng+ixg, ny+iyg)] = 0.0;
                    }
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            npy_intp xminymax_ipatch = xmin_index[ymax_ipatch];
            if (xminymax_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(ixg, ny-ng+iyg)] += field[xminymax_ipatch][INDEX2(nx+ixg, -ng+iyg)];
                        field[xminymax_ipatch][INDEX2(nx+ixg, -ng+iyg)] = 0.0;
                    }
                }
            }
            
            npy_intp xmaxymax_ipatch = xmax_index[ymax_ipatch];
            if (xmaxymax_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(nx-ng+ixg, ny-ng+iyg)] += field[xmaxymax_ipatch][INDEX2(-ng+ixg, -ng+iyg)];
                        field[xmaxymax_ipatch][INDEX2(-ng+ixg, -ng+iyg)] = 0.0;
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

static PyObject* sync_guard_fields(PyObject* self, PyObject* args) {
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

    // Get field data pointers for E and B fields
    double **ex = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "ex");
    double **ey = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "ey");
    double **ez = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "ez");
    double **bx = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "bx");
    double **by = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "by");
    double **bz = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "bz");

    const npy_intp* xmin_index = (npy_intp*)PyArray_DATA(xmin_index_arr);
    const npy_intp* xmax_index = (npy_intp*)PyArray_DATA(xmax_index_arr);
    const npy_intp* ymin_index = (npy_intp*)PyArray_DATA(ymin_index_arr);
    const npy_intp* ymax_index = (npy_intp*)PyArray_DATA(ymax_index_arr);

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

        npy_intp xmin_ipatch = xmin_index[ipatch];
        npy_intp xmax_ipatch = xmax_index[ipatch];
        npy_intp ymin_ipatch = ymin_index[ipatch];
        npy_intp ymax_ipatch = ymax_index[ipatch];

        // X direction sync
        if (xmin_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    field[ipatch][INDEX2(-ixg-1, iy)] = field[xmin_ipatch][INDEX2(nx-ng+ixg, iy)];
                }
            }
        }
        
        if (xmax_ipatch >= 0) {
            for (npy_intp ixg = 0; ixg < ng; ixg++) {
                for (npy_intp iy = 0; iy < ny; iy++) {
                    field[ipatch][INDEX2(nx+ixg, iy)] = field[xmax_ipatch][INDEX2(ixg, iy)];
                }
            }
        }

        // Y direction sync
        if (ymin_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    field[ipatch][INDEX2(ix, -iyg-1)] = field[ymin_ipatch][INDEX2(ix, ny-ng+iyg)];
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            for (npy_intp ix = 0; ix < nx; ix++) {
                for (npy_intp iyg = 0; iyg < ng; iyg++) {
                    field[ipatch][INDEX2(ix, ny+iyg)] = field[ymax_ipatch][INDEX2(ix, iyg)];
                }
            }
        }

        // Corner synchronization
        if (ymin_ipatch >= 0) {
            npy_intp xminymin_ipatch = xmin_index[ymin_ipatch];
            if (xminymin_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(-ixg-1, -iyg-1)] = field[xminymin_ipatch][INDEX2(nx-ng+ixg, ny-ng+iyg)];
                    }
                }
            }
            
            npy_intp xmaxymin_ipatch = xmax_index[ymin_ipatch];
            if (xmaxymin_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(nx+ixg, -iyg-1)] = field[xmaxymin_ipatch][INDEX2(ixg, ny-ng+iyg)];
                    }
                }
            }
        }
        
        if (ymax_ipatch >= 0) {
            npy_intp xminymax_ipatch = xmin_index[ymax_ipatch];
            if (xminymax_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(-ixg-1, ny+iyg)] = field[xminymax_ipatch][INDEX2(nx-ng+ixg, iyg)];
                    }
                }
            }
            
            npy_intp xmaxymax_ipatch = xmax_index[ymax_ipatch];
            if (xmaxymax_ipatch >= 0) {
                for (npy_intp ixg = 0; ixg < ng; ixg++) {
                    for (npy_intp iyg = 0; iyg < ng; iyg++) {
                        field[ipatch][INDEX2(nx+ixg, ny+iyg)] = field[xmaxymax_ipatch][INDEX2(ixg, iyg)];
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    // Cleanup
    free(ex); free(ey); free(ez);
    free(bx); free(by); free(bz);
    Py_DECREF(fields_list);
    
    Py_RETURN_NONE;
}

static PyObject* sync_currents_3d(PyObject* self, PyObject* args) {
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

    double **jx = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "jx");
    double **jy = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "jy"); 
    double **jz = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "jz");
    double **rho = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "rho");

    const npy_intp* xmin_index = (npy_intp*)PyArray_DATA(xmin_index_arr);
    const npy_intp* xmax_index = (npy_intp*)PyArray_DATA(xmax_index_arr);
    const npy_intp* ymin_index = (npy_intp*)PyArray_DATA(ymin_index_arr);
    const npy_intp* ymax_index = (npy_intp*)PyArray_DATA(ymax_index_arr);
    const npy_intp* zmin_index = (npy_intp*)PyArray_DATA(zmin_index_arr);
    const npy_intp* zmax_index = (npy_intp*)PyArray_DATA(zmax_index_arr);

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

        npy_intp xmin_ipatch = xmin_index[ipatch];
        npy_intp xmax_ipatch = xmax_index[ipatch];
        npy_intp ymin_ipatch = ymin_index[ipatch];
        npy_intp ymax_ipatch = ymax_index[ipatch];
        npy_intp zmin_ipatch = zmin_index[ipatch];
        npy_intp zmax_ipatch = zmax_index[ipatch];

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
        if (xmin_ipatch >= 0) {
            SYNC_BOUNDARY_3D(xmin_ipatch, 
                0, nx, ng, 
                0, 0, ny, 
                0, 0, nz
            )
        }
        
        if (xmax_ipatch >= 0) {
            SYNC_BOUNDARY_3D(xmax_ipatch, 
                nx-ng, -ng, ng, 
                0, 0, ny, 
                0, 0, nz
            )
        }

        // Y direction sync
        if (ymin_ipatch >= 0) {
            SYNC_BOUNDARY_3D(ymin_ipatch, 
                0, 0, nx, 
                0, ny, ng, 
                0, 0, nz
            )
        }
        
        if (ymax_ipatch >= 0) {
            SYNC_BOUNDARY_3D(ymax_ipatch, 
                0, 0, nx, 
                ny-ng, -ng, ng, 
                0, 0, nz
            )
        }

        // Z direction sync
        if (zmin_ipatch >= 0) {
            SYNC_BOUNDARY_3D(zmin_ipatch, 
                0, 0, nx, 
                0, 0, ny, 
                0, nz, ng
            )
        }
        
        if (zmax_ipatch >= 0) {
            SYNC_BOUNDARY_3D(zmax_ipatch, 
                0, 0, nx, 
                0, 0, ny, 
                nz-ng, -ng, ng
            )
        }

        // Edge synchronization (3D)
        if (xmin_ipatch >= 0) {
            // Handle x-min y-min edge
            npy_intp xminymin_ipatch = ymin_index[xmin_ipatch];
            if (xminymin_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xminymin_ipatch, 
                    0, nx, ng, 
                    0, ny, ng, 
                    0, 0, nz
                )
            }
            
            // Handle x-min y-max edge
            npy_intp xminymax_ipatch = ymax_index[xmin_ipatch];
            if (xminymax_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xminymax_ipatch, 
                    0, nx, ng, 
                    ny-ng, -ng, ng, 
                    0, 0, nz
                )
            }

            // Handle x-min z-min edge
            npy_intp xminzmin_ipatch = zmin_index[xmin_ipatch];
            if (xminzmin_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xminzmin_ipatch, 
                    0, nx, ng, 
                    0, 0, ny, 
                    0, nz, ng
                )
            }

            // Handle x-min z-max edge
            npy_intp xminzmax_ipatch = zmax_index[xmin_ipatch];
            if (xminzmax_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xminzmax_ipatch, 
                    0, nx, ng, 
                    0, 0, ny, 
                    nz-ng, -ng, ng
                )
            }
        }

        if (xmax_ipatch >= 0) {
            // Handle xmax y-min edge
            npy_intp xmaxymin_ipatch = ymin_index[xmax_ipatch];
            if (xmaxymin_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xmaxymin_ipatch,
                    nx-ng, -ng, ng,
                    0, ny, ng,
                    0, 0, nz
                )
            }
            
            // Handle xmax y-max edge
            npy_intp xmaxymax_ipatch = ymax_index[xmax_ipatch];
            if (xmaxymax_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xmaxymax_ipatch,
                    nx-ng, -ng, ng,
                    ny-ng, -ng, ng,
                    0, 0, nz
                )
            }

            // Handle xmax z-min edge
            npy_intp xmaxzmin_ipatch = zmin_index[xmax_ipatch];
            if (xmaxzmin_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xmaxzmin_ipatch,
                    nx-ng, -ng, ng,
                    0, 0, ny,
                    0, nz, ng
                )
            }

            // Handle xmax z-max edge
            npy_intp xmaxzmax_ipatch = zmax_index[xmax_ipatch];
            if (xmaxzmax_ipatch >= 0) {
                SYNC_BOUNDARY_3D(xmaxzmax_ipatch,
                    nx-ng, -ng, ng,
                    0, 0, ny,
                    nz-ng, -ng, ng
                )
            }
        }

        if (ymin_ipatch >= 0) {
            // Handle y-min z-min edge
            npy_intp yminzmin_ipatch = zmin_index[ymin_ipatch];
            if (yminzmin_ipatch >= 0) {
                SYNC_BOUNDARY_3D(yminzmin_ipatch,
                    0, 0, nx,
                    0, ny, ng,
                    0, nz, ng
                )
            }

            // Handle y-min z-max edge
            npy_intp yminzmax_ipatch = zmax_index[ymin_ipatch];
            if (yminzmax_ipatch >= 0) {
                SYNC_BOUNDARY_3D(yminzmax_ipatch,
                    0, 0, nx,
                    0, ny, ng,
                    nz-ng, -ng, ng
                )
            }
        }

        if (ymax_ipatch >= 0) {
            // Handle y-max z-min edge
            npy_intp ymaxzmin_ipatch = zmin_index[ymax_ipatch];
            if (ymaxzmin_ipatch >= 0) {
                SYNC_BOUNDARY_3D(ymaxzmin_ipatch,
                    0, 0, nx,
                    ny-ng, -ng, ng,
                    0, nz, ng
                )
            }

            // Handle y-max z-max edge
            npy_intp ymaxzmax_ipatch = zmax_index[ymax_ipatch];
            if (ymaxzmax_ipatch >= 0) {
                SYNC_BOUNDARY_3D(ymaxzmax_ipatch,
                    0, 0, nx,
                    ny-ng, -ng, ng,
                    nz-ng, -ng, ng
                )
            }
        }

        // Corner synchronization (3D)
        // xmin ymin zmin
        if (xmin_ipatch >= 0 && ymin_index[xmin_ipatch] >= 0 && zmin_index[ymin_index[xmin_ipatch]] >= 0) {
            npy_intp xminyminzmin_ipatch = zmin_index[ymin_index[xmin_ipatch]];
            SYNC_BOUNDARY_3D(xminyminzmin_ipatch,
                0, nx, ng,
                0, ny, ng,
                0, nz, ng
            )
        }

        // xmin ymin zmax
        if (xmin_ipatch >= 0 && ymin_index[xmin_ipatch] >= 0 && zmax_index[ymin_index[xmin_ipatch]] >= 0) {
            npy_intp xminyminzmax_ipatch = zmax_index[ymin_index[xmin_ipatch]];
            SYNC_BOUNDARY_3D(xminyminzmax_ipatch,
                0, nx, ng,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        // xmin ymax zmin
        if (xmin_ipatch >= 0 && ymax_index[xmin_ipatch] >= 0 && zmin_index[ymax_index[xmin_ipatch]] >= 0) {
            npy_intp xminymaxzmin_ipatch = zmin_index[ymax_index[xmin_ipatch]];
            SYNC_BOUNDARY_3D(xminymaxzmin_ipatch,
                0, nx, ng,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        // xmin ymax zmax
        if (xmin_ipatch >= 0 && ymax_index[xmin_ipatch] >= 0 && zmax_index[ymax_index[xmin_ipatch]] >= 0) {
            npy_intp xminymaxzmax_ipatch = zmax_index[ymax_index[xmin_ipatch]];
            SYNC_BOUNDARY_3D(xminymaxzmax_ipatch,
                0, nx, ng,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

        // xmax ymin zmin
        if (xmax_ipatch >= 0 && ymin_index[xmax_ipatch] >= 0 && zmin_index[ymin_index[xmax_ipatch]] >= 0) {
            npy_intp xmaxyminzmin_ipatch = zmin_index[ymin_index[xmax_ipatch]];
            SYNC_BOUNDARY_3D(xmaxyminzmin_ipatch,
                nx-ng, -ng, ng,
                0, ny, ng,
                0, nz, ng
            )
        }

        // xmax ymin zmax
        if (xmax_ipatch >= 0 && ymin_index[xmax_ipatch] >= 0 && zmax_index[ymin_index[xmax_ipatch]] >= 0) {
            npy_intp xmaxyminzmax_ipatch = zmax_index[ymin_index[xmax_ipatch]];
            SYNC_BOUNDARY_3D(xmaxyminzmax_ipatch,
                nx-ng, -ng, ng,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        // xmax ymax zmin
        if (xmax_ipatch >= 0 && ymax_index[xmax_ipatch] >= 0 && zmin_index[ymax_index[xmax_ipatch]] >= 0) {
            npy_intp xmaxymaxzmin_ipatch = zmin_index[ymax_index[xmax_ipatch]];
            SYNC_BOUNDARY_3D(xmaxymaxzmin_ipatch,
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        // xmax ymax zmax
        if (xmax_ipatch >= 0 && ymax_index[xmax_ipatch] >= 0 && zmax_index[ymax_index[xmax_ipatch]] >= 0) {
            npy_intp xmaxymaxzmax_ipatch = zmax_index[ymax_index[xmax_ipatch]];
            SYNC_BOUNDARY_3D(xmaxymaxzmax_ipatch,
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

    }
    Py_END_ALLOW_THREADS

    free(jx); free(jy); free(jz); free(rho);
    Py_DECREF(fields_list);
    
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

    double **ex = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "ex");
    double **ey = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "ey");
    double **ez = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "ez");
    double **bx = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "bx");
    double **by = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "by");
    double **bz = GET_ATTR_DOUBLEARRAY(fields_list, npatches, "bz");

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
    {"sync_currents", sync_currents, METH_VARARGS, "Synchronize currents between patches (2D)"},
    {"sync_guard_fields", sync_guard_fields, METH_VARARGS, "Synchronize guard cells between patches (2D)"},
    {"sync_currents_3d", sync_currents_3d, METH_VARARGS, "Synchronize currents between patches (3D)"},
    {"sync_guard_fields_3d", sync_guard_fields_3d, METH_VARARGS, "Synchronize guard cells between patches (3D)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "sync_fields",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_sync_fields(void) {
    import_array();
    return PyModule_Create(&module);
}
