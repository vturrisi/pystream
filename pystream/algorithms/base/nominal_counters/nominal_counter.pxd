cimport cython


ctypedef fused value_type:
    cython.numeric
    cython.p_char
    cython.char
    unicode


cdef class NominalCounter:
    cdef public int total
    cdef int _n_classes
    cdef tuple _values
    cdef dict _counts, _totals

    cdef double _memory_size(self)

    cdef _add_instance(self, object value, int y, int weight)

    cdef int _get_count(self, object value)

    cdef tuple _get_values(self)

    cdef bint _empty(self)

    cdef dict _get_distribution(self)

    cdef dict _get_total_per_value(self)

    cdef void _merge(self, NominalCounter other)

    cdef double _get_proba(self, object value, int y)

    cdef dict _get_probas(self)

    cdef tuple _possible_values(self)
