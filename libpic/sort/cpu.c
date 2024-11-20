#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

typedef npy_intp intp;

/**
 * Calculate the cell index for each particle based on its position.
 * 
 * @param x Pointer to the array of particle x-coordinates.
 * @param y Pointer to the array of particle y-coordinates.
 * @param is_dead Pointer to the array indicating if a particle is dead.
 * @param npart Number of particles.
 * @param nx Number of cells in the x-direction.
 * @param ny Number of cells in the y-direction.
 * @param dx Cell size in the x-direction.
 * @param dy Cell size in the y-direction.
 * @param x0 Origin x-coordinate.
 * @param y0 Origin y-coordinate.
 * @param particle_cell_indices Pointer to the array to store the cell indices.
 * @param grid_cell_count Pointer to the array to store the count of particles in each cell.
 */
static void calculate_cell_index(
    double* x, double* y, npy_bool* is_dead, 
    intp npart, intp nx, intp ny, double dx, double dy, double x0, double y0, 
    intp* particle_cell_indices, intp* grid_cell_count
) {
    intp ix, iy, ip, icell;
    icell = 0;
    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (intp) floor((x[ip] - x0) / dx);  // Calculate the x-index of the cell
            iy = (intp) floor((y[ip] - y0) / dy);  // Calculate the y-index of the cell
            icell = iy + ix * ny;  // Calculate the cell index
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny) {
                particle_cell_indices[ip] = icell;  // Store the cell index for the particle
                grid_cell_count[icell] += 1;  // Increment the count of particles in the cell
            }
        } else {
            particle_cell_indices[ip] = -1;  // Mark dead particles with a cell index of -1
        }
    }
}

/**
 * Python wrapper for the calculate_cell_index function.
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return None.
 */
static PyObject* _calculate_cell_index(PyObject* self, PyObject* args) {
    PyObject *x, *y, *is_dead, *particle_cell_indices, *grid_cell_count;
    intp nx, ny, npart;
    double dx, dy, x0, y0;
    if (!PyArg_ParseTuple(args, "OOOnnnddddOO", 
        &x, &y, &is_dead, 
        &npart, &nx, &ny, &dx, &dy, &x0, &y0, 
        &particle_cell_indices, &grid_cell_count)) {
        return NULL;  
    }

    calculate_cell_index(
        (double*) PyArray_DATA(x), (double*) PyArray_DATA(y), (npy_bool*) PyArray_DATA(is_dead), 
        npart, nx, ny, dx, dy, x0, y0, 
        (intp*) PyArray_DATA(particle_cell_indices), (intp*) PyArray_DATA(grid_cell_count)
    );
    Py_RETURN_NONE;  
}

/**
 * Perform a cycle sort to sort particles within cells.
 * 
 * @param cell_bound_min Pointer to the array of minimum bounds for each cell.
 * @param cell_bound_max Pointer to the array of maximum bounds for each cell.
 * @param nx Number of cells in the x-direction.
 * @param ny Number of cells in the y-direction.
 * @param particle_cell_indices Pointer to the array of particle cell indices.
 * @param is_dead Pointer to the array indicating if a particle is dead.
 * @param sorted_idx Pointer to the array to store the sorted indices.
 * @return Number of operations performed.
 */
static intp cycle_sort(
    intp* cell_bound_min, intp* cell_bound_max, 
    intp nx, intp ny, 
    intp* particle_cell_indices, npy_bool* is_dead, intp* sorted_idx
) {
    intp ops = 0;
    intp ix, iy, ip, ip_src, ip_dst, icell_src, icell_dst, idx_dst;
    
    for (ix = 0; ix < nx; ix++) {
        for (iy = 0; iy < ny; iy++) {
            icell_src = iy + ix * ny;  // Calculate the source cell index
            for (ip = cell_bound_min[icell_src]; ip < cell_bound_max[icell_src]; ip++) {
                if (is_dead[ip]) {
                    continue;  // Skip dead particles
                }
                if (particle_cell_indices[ip] == icell_src) {
                    continue;  // Skip particles already in the correct cell
                }
                ip_src = ip;
                icell_dst = particle_cell_indices[ip_src];  // Get the destination cell index
                idx_dst = sorted_idx[ip_src];  // Get the destination index

                while (icell_dst != icell_src) {
                    for (ip_dst = cell_bound_min[icell_dst]; ip_dst < cell_bound_max[icell_dst]; ip_dst++) {
                        if (particle_cell_indices[ip_dst] != icell_dst || is_dead[ip_dst]) {
                            // swap
                            intp tmp = particle_cell_indices[ip_dst];
                            particle_cell_indices[ip_dst] = icell_dst;
                            icell_dst = tmp;

                            tmp = sorted_idx[ip_dst];
                            sorted_idx[ip_dst] = idx_dst;
                            idx_dst = tmp;
                            
                            ip_src = ip_dst;  // Update source index
                            ops += 1;  // Increment operation count
                            break;
                        }
                    }
                    if (is_dead[ip_dst]) {
                        break;  // Break if the destination particle is dead
                    }
                }
                particle_cell_indices[ip] = icell_dst;  // Update particle cell index
                sorted_idx[ip] = idx_dst;  // Update sorted index
                if (is_dead[ip_dst]) {
                    is_dead[ip] = 1;  // Mark source particle as dead
                    is_dead[ip_dst] = 0;  // Mark destination particle as alive
                }
            }
        }
    }
    return ops;  // Return the number of operations performed
}

/**
 * Python wrapper for the cycle_sort function.
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return Number of operations performed.
 */
static PyObject* _cycle_sort(PyObject* self, PyObject* args) {
    PyArrayObject *cell_bound_min, *cell_bound_max, *particle_cell_indices, *is_dead, *sorted_idx;
    intp nx, ny;
    if (!PyArg_ParseTuple(args, "OOnnOOO", 
        &cell_bound_min, &cell_bound_max, 
        &nx, &ny, 
        &particle_cell_indices, &is_dead, &sorted_idx)) {
        return NULL;  
    }
    intp ops = cycle_sort(
        (intp*) PyArray_DATA(cell_bound_min), (intp*) PyArray_DATA(cell_bound_max), 
        nx, ny, 
        (intp*) PyArray_DATA(particle_cell_indices), (npy_bool*) PyArray_DATA(is_dead), (intp*) PyArray_DATA(sorted_idx)
    );
    return PyLong_FromLong(ops);  
}

/**
 * Calculate the bounds for each cell based on the particle counts.
 * 
 * @param grid_cell_count Pointer to the array of particle counts in each cell.
 * @param cell_bound_min Pointer to the array to store the minimum bounds.
 * @param cell_bound_max Pointer to the array to store the maximum bounds.
 * @param nx Number of cells in the x-direction.
 * @param ny Number of cells in the y-direction.
 */
static void sorted_cell_bound(
    intp* grid_cell_count, intp* cell_bound_min, intp* cell_bound_max, 
    intp nx, intp ny
) {
    intp icell, icell_prev;
    cell_bound_min[0] = 0;  // Initialize the minimum bound for the first cell

    for (icell = 1; icell < nx * ny; icell++) {
        icell_prev = icell - 1;  // Get the previous cell index
        cell_bound_min[icell] = cell_bound_min[icell_prev] + grid_cell_count[icell_prev];  // Calculate the minimum bound for the current cell
        cell_bound_max[icell_prev] = cell_bound_min[icell];  // Calculate the maximum bound for the previous cell
    }
    cell_bound_max[nx * ny - 1] = cell_bound_min[nx * ny - 1] + grid_cell_count[nx * ny - 1];  // Calculate the maximum bound for the last cell
}

/**
 * Python wrapper for sorted_cell_bound
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return None.
 */
static PyObject* _sorted_cell_bound(PyObject* self, PyObject* args) {
    PyArrayObject *grid_cell_count, *cell_bound_min, *cell_bound_max;
    intp nx, ny;
    if (!PyArg_ParseTuple(args, "OOOnn", 
        &grid_cell_count, &cell_bound_min, &cell_bound_max, 
        &nx, &ny)) {
        return NULL;  
    }
    sorted_cell_bound(
        (intp*) PyArray_DATA(grid_cell_count), (intp*) PyArray_DATA(cell_bound_min), (intp*) PyArray_DATA(cell_bound_max), 
        nx, ny
    );
    Py_RETURN_NONE;  
}


/**
 * Sort particles in patches using OpenMP for parallel execution.
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return None.
 */
static PyObject* sort_particles_patches(PyObject* self, PyObject* args) {
    PyObject *grid_cell_count_list, *cell_bound_min_list, *cell_bound_max_list, *x0s, *y0s, *particle_cell_indices_list, *sorted_indices_list, *x_list, *y_list, *is_dead_list, *attrs_list;
    intp nx, ny, npatches;
    double dx, dy;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOndd", 
        &grid_cell_count_list, &cell_bound_min_list, &cell_bound_max_list, 
        &x0s, &y0s, 
        &nx, &ny, &dx, &dy, 
        &npatches, 
        &particle_cell_indices_list, &sorted_indices_list, &x_list, &y_list, &is_dead_list, &attrs_list)) {
        return NULL;  // Return NULL if argument parsing fails
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;  // Return None if there are no patches
    }

    intp** grid_cell_count_data = malloc(npatches * sizeof(intp*));  // Allocate memory for grid cell count data
    intp** cell_bound_min_data = malloc(npatches * sizeof(intp*));  // Allocate memory for cell bound min data
    intp** cell_bound_max_data = malloc(npatches * sizeof(intp*));  // Allocate memory for cell bound max data
    double* x0_data = malloc(npatches * sizeof(double));  // Allocate memory for x0 data
    double* y0_data = malloc(npatches * sizeof(double));  // Allocate memory for y0 data
    intp** particle_cell_indices_data = malloc(npatches * sizeof(intp*));  // Allocate memory for particle cell indices data
    intp** sorted_indices_data = malloc(npatches * sizeof(intp*));  // Allocate memory for sorted indices data
    double** x_data = malloc(npatches * sizeof(double*));  // Allocate memory for x data
    double** y_data = malloc(npatches * sizeof(double*));  // Allocate memory for y data
    npy_bool** is_dead_data = malloc(npatches * sizeof(npy_bool*));  // Allocate memory for is_dead data

    intp nattrs = PyList_Size(attrs_list) / npatches;  
    double** attrs_data = malloc(npatches * nattrs * sizeof(double*));  // Allocate memory for attrs data

    intp* npart_data = malloc(npatches * sizeof(intp));  

    for (intp ipatch = 0; ipatch < npatches; ipatch++) {
        grid_cell_count_data[ipatch] = (intp*)PyArray_DATA((PyArrayObject*)PyList_GetItem(grid_cell_count_list, ipatch));  // Get grid cell count data
        cell_bound_min_data[ipatch] = (intp*)PyArray_DATA((PyArrayObject*)PyList_GetItem(cell_bound_min_list, ipatch));  // Get cell bound min data
        cell_bound_max_data[ipatch] = (intp*)PyArray_DATA((PyArrayObject*)PyList_GetItem(cell_bound_max_list, ipatch));  // Get cell bound max data
        x0_data[ipatch] = PyFloat_AsDouble(PyList_GetItem(x0s, ipatch));  // Get x0 data
        y0_data[ipatch] = PyFloat_AsDouble(PyList_GetItem(y0s, ipatch));  // Get y0 data
        particle_cell_indices_data[ipatch] = (intp*)PyArray_DATA((PyArrayObject*)PyList_GetItem(particle_cell_indices_list, ipatch));  // Get particle cell indices data
        sorted_indices_data[ipatch] = (intp*)PyArray_DATA((PyArrayObject*)PyList_GetItem(sorted_indices_list, ipatch));  // Get sorted indices data
        x_data[ipatch] = (double*)PyArray_DATA((PyArrayObject*)PyList_GetItem(x_list, ipatch));  // Get x data
        y_data[ipatch] = (double*)PyArray_DATA((PyArrayObject*)PyList_GetItem(y_list, ipatch));  // Get y data
        is_dead_data[ipatch] = (npy_bool*)PyArray_DATA((PyArrayObject*)PyList_GetItem(is_dead_list, ipatch));  // Get is_dead data

        npart_data[ipatch] = PyArray_DIM((PyArrayObject*)PyList_GetItem(is_dead_list, ipatch), 0);
        for (intp iattr = 0; iattr < nattrs; iattr++) {
            attrs_data[ipatch * nattrs + iattr] = (double*)PyArray_DATA((PyArrayObject*)PyList_GetItem(attrs_list, ipatch * nattrs + iattr));  
        }
    }

    #pragma omp parallel for  // Parallelize the loop using OpenMP
    for (intp ipatch = 0; ipatch < npatches; ipatch++) {
        intp* grid_cell_count = grid_cell_count_data[ipatch];  // Get grid cell count for the current patch
        intp* cell_bound_min = cell_bound_min_data[ipatch];  // Get cell bound min for the current patch
        intp* cell_bound_max = cell_bound_max_data[ipatch];  // Get cell bound max for the current patch
        double x0 = x0_data[ipatch];  // Get x0 for the current patch
        double y0 = y0_data[ipatch];  // Get y0 for the current patch
        intp* particle_cell_indices = particle_cell_indices_data[ipatch];  // Get particle cell indices for the current patch
        intp* sorted_indices = sorted_indices_data[ipatch];  // Get sorted indices for the current patch
        double* x = x_data[ipatch];  // Get x for the current patch
        double* y = y_data[ipatch];  // Get y for the current patch
        npy_bool* is_dead = is_dead_data[ipatch];  // Get is_dead for the current patch
        intp npart = npart_data[ipatch];  


        for (intp icell = 0; icell < nx * ny; icell++) {
            grid_cell_count[icell] = 0;  // Initialize grid cell count for the current patch
        }

        calculate_cell_index(
            x, y, is_dead,
            npart,
            nx, ny, dx, dy,
            x0, y0,
            particle_cell_indices, grid_cell_count
        );  // Calculate cell indices for the current patch

        sorted_cell_bound(
            grid_cell_count, 
            cell_bound_min, 
            cell_bound_max, 
            nx, ny
        );  // Calculate cell bounds for the current patch

        for (intp ip = 0; ip < npart; ip++) {
            sorted_indices[ip] = ip;  // Initialize sorted indices for the current patch
        }

        cycle_sort(cell_bound_min, cell_bound_max, nx, ny, particle_cell_indices, is_dead, sorted_indices);  // Perform cycle sort for the current patch

        double* buf = (double*) malloc(npart * sizeof(double));  // Allocate memory for buffer
        for (intp iattr = 0; iattr < nattrs; iattr++) {
            double* attr = attrs_data[ipatch * nattrs + iattr];  
            for (intp ip = 0; ip < npart; ip++) {
                if (ip != sorted_indices[ip]) {
                    buf[ip] = attr[sorted_indices[ip]];  // Copy attributes to buffer
                }
            }
            for (intp ip = 0; ip < npart; ip++) {
                if (ip != sorted_indices[ip]) {
                    attr[ip] = buf[ip];  // Copy attributes from buffer
                }
            }
        }
        free(buf);  // Free buffer memory
    }

    free(grid_cell_count_data);
    free(cell_bound_min_data);
    free(cell_bound_max_data);
    free(x0_data);
    free(y0_data);
    free(particle_cell_indices_data);
    free(sorted_indices_data);
    free(x_data);
    free(y_data);
    free(is_dead_data);
    free(attrs_data);
    free(npart_data);

    Py_RETURN_NONE;
}

static PyMethodDef SortMethods[] = {
    {"sort_particles_patches", sort_particles_patches, METH_VARARGS, "Sort particles patches"},
    {"_calculate_cell_index", _calculate_cell_index, METH_VARARGS, "Calculate cell index"},
    {"_sorted_cell_bound", _sorted_cell_bound, METH_VARARGS, "Calculate sorted cell bound"},
    {"_cycle_sort", _cycle_sort, METH_VARARGS, "Cycle sort"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sortmodule = {
    PyModuleDef_HEAD_INIT,
    "cpu",
    NULL,
    -1,
    SortMethods
};

PyMODINIT_FUNC PyInit_cpu(void) {
    import_array();
    return PyModule_Create(&sortmodule);
}
