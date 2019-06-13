cimport numpy as np
from vfdt cimport Node, VFDT


cdef class BoostNode(Node):
    cdef:
        public np.ndarray _dist_pred
        public list _nb_estimators

    # cdef _update_attr_estimators(self, np.ndarray X, int y, int weight)

    cdef double _compute_lambda_weight(self, double prob,
                                       double min_l, double max_l)

    cdef void _update_nb_estimators(self, np.ndarray X, int y, int weight)

    cdef void learn_from_instance_boost(self, np.ndarray X, int y,
                                        int weight, int min_l, int max_l)


cdef class OLBoostVFDT(VFDT):
    cdef int _min_l, _max_l
