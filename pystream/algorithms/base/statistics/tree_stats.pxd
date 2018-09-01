cdef class TreeStats:
    cdef int _n_classes
    cdef dict _stats

    cdef double _memory_size(self)
