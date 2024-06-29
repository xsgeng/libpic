# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
from cython cimport bool
import numpy as np
cimport numpy as cnp


from libc.stdlib cimport malloc, free

cdef struct ArrayInfo:
    void* data_ptr
    cnp.npy_intp* shape
    int ndim
    
    
cdef class CListBase(list):

    def __init__(self, list arrays):
        assert all(arr.dtype == arrays[0].dtype for arr in arrays), "arrays types are not same"
        self.dtype = cnp.PyArray_TYPE(arrays[0])
        super().__init__(arrays)
        
        self.size = len(arrays)
        self.array_infos = <ArrayInfo*>malloc(self.size * sizeof(ArrayInfo))
        if not self.array_infos:
            raise MemoryError()

        cdef int i
        for i in range(self.size):
            arr = arrays[i]
            if not isinstance(arr, np.ndarray):
                raise ValueError("All elements must be NumPy arrays")
            self.array_infos[i].data_ptr = <void*>cnp.PyArray_DATA(arr)
            self.array_infos[i].shape = <cnp.npy_intp*>cnp.PyArray_DIMS(arr)
            self.array_infos[i].ndim = cnp.PyArray_NDIM(arr)

    def __setitem__(self, i, arr):
        super().__setitem__(i, arr)
        if not isinstance(arr, np.ndarray):
            raise ValueError("All elements must be NumPy arrays")
        self.array_infos[i].data_ptr = <void*>cnp.PyArray_DATA(arr)
        self.array_infos[i].shape = <cnp.npy_intp*>cnp.PyArray_DIMS(arr)
        self.array_infos[i].ndim = cnp.PyArray_NDIM(arr)

    def __dealloc__(self):
        if self.array_infos is not NULL:
            free(self.array_infos)

    cdef cnp.npy_intp* get_shape(self, int index) noexcept nogil:
        if index < 0 or index >= self.size:
            return NULL
        return self.array_infos[index].shape


    cdef cnp.npy_intp get_size(self, int index) noexcept nogil:
        if index < 0 or index >= self.size:
            return 0
        cdef cnp.npy_intp size, idim
        
        size = 1
        for idim in range(self.array_infos[index].ndim):
            size *= self.array_infos[index].shape[idim]
            
        return size

class CList:
    def __new__(self, arrays):
        assert all(arr.dtype == arrays[0].dtype for arr in arrays), "arrays types are not same"
        if arrays[0].dtype == np.float64:
            return CListDouble(arrays)
        elif arrays[0].dtype == bool:
            return CListBool(arrays)
        elif arrays[0].dtype == int:
            return CListIntp(arrays)
        else:
            raise TypeError("Unsupported dtype")