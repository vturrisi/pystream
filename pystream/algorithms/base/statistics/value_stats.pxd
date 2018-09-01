cdef class SimpleStats:
    cdef dict _stats

    cdef double _memory_size(self)

    cdef void _update(self, double value)

    cdef void _remove(self, double value)


cdef class NormalDistValueStats(SimpleStats):

    cdef double _mean_plus_factor_std(self, double factor)
