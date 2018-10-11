from vfdt cimport VFDT, Node
from statistics.value_stats cimport NormalDistValueStats


cdef class SVFDT(VFDT):
    cdef:
        double _imp_factor, _gain_factor, _n_factor
        NormalDistValueStats _imp_stats, _gain_stats, _n_stats

    cdef double _min_entropy_leaves(self, double factor)

    cdef double _min_gini_leaves(self, double factor)

    cdef bint _can_split_svfdt(self, Node leaf, double best_gain)
