cimport numpy as np


cdef class ClassGaussianEstimator:
    """
    Estimates the distribution of values (according to their class label)
    assuming that they follow the normal distribution
    """

    cdef public int total
    cdef public double max_value, min_value
    cdef int _n_classes, _n_bins
    cdef list _estimators
    cdef np.ndarray _max_values, _min_values

    cdef double _memory_size(self)

    cdef _add_instance(self, double value, int y, int weight)

    cdef np.ndarray[np.double_t, ndim=2] _get_le_gt(self, double split_value)

    cdef double _get_proba(self, double value, int y)

    cdef void _merge(self, ClassGaussianEstimator other)


cdef class GaussianEstimator:
    """
    Used as helper class to the ClassGaussianNumericEstimator
    This class constructs normal distribution from values
    in an online fashion.
    """

    cdef public int _n
    cdef public double _mean, _variance_sum, _variance, _stdev

    cdef double memory_size(self)

    cdef add_value(self, double value, int weight)

    cpdef merge(self, GaussianEstimator other)

    cdef double proba_density(self, double value)

    cdef tuple estimate_le_gt_distribution(self, double split_value)

    cdef double normal_probability(self, double a)

    cdef double error_function(self, double x)

    cdef double error_function_complemented(self, double a)

    cdef double polevl(self, double x, double[:] coef, int N)

    cdef double p1evl(self, double x, double[:] coef, int N)
