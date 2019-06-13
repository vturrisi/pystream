from vfdt cimport Node
from olboost_svfdt cimport OLBoostSVFDT


cdef class OLBoostSVFDT_II(OLBoostSVFDT):
    cpdef bint _can_split_svfdt(self, Node leaf, double best_gain)
