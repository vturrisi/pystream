from vfdt cimport Node
from olboost_vfdt cimport OLBoostVFDT
from statistics.value_stats cimport NormalDistValueStats


cdef class OLBoostSVFDT(OLBoostVFDT):
    cdef double _imp_factor, _gain_factor, _n_factor
    cdef NormalDistValueStats _imp_stats, _gain_stats, _n_stats

    cpdef double memory_size(self)

    cdef double _min_entropy_leaves(self, double factor)

    cdef double _min_gini_leaves(self, double factor)

    cpdef bint _can_split_svfdt(self, Node leaf, double best_gain)
