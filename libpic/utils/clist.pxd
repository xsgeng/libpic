from cython cimport bool
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

ctypedef cnp.npy_intp intp

cdef struct ArrayInfo:
    void* data_ptr
    intp* shape
    int ndim


cdef class CListBase(list):
    cdef ArrayInfo* array_infos
    cdef int size
    cdef int dtype
    cdef intp* get_shape(self, int index) noexcept nogil
    cdef intp get_size(self, int index) noexcept nogil


cdef class CListDouble(CListBase):
    cdef inline double* get_ptr(self, int index) noexcept nogil:
        if index < 0 or index >= self.size:
            return NULL

        return <double*>self.array_infos[index].data_ptr

cdef class CListBool(CListBase):
    cdef inline cnp.npy_bool* get_ptr(self, int index) noexcept nogil:
        if index < 0 or index >= self.size:
            return NULL

        return <cnp.npy_bool*>self.array_infos[index].data_ptr

cdef class CListIntp(CListBase):
     cdef inline intp* get_ptr(self, int index) noexcept nogil:
        if index < 0 or index >= self.size:
            return NULL

        return <intp*>self.array_infos[index].data_ptr