# cython: boundscheck=False, wraparound=False

from cpython cimport array
cimport numpy as np
import numpy as np
import array
from sys import getsizeof
from typing import Generator
from libc.math cimport sqrt, pi, exp, pow
cimport cython


cConstants_NORMAL_CONST = sqrt(2 * pi)
cConstants_SQRTH = 7.07106781186547524401E-1
cConstants_MAXLOG = 7.09782712893383996732E2

cdef array.array Constants_T = array.array('d',
                                           [9.60497373987051638749E0,
                                            9.00260197203842689217E1,
                                            2.23200534594684319226E3,
                                            7.00332514112805075473E3,
                                            5.55923013010394962768E4])

cdef array.array Constants_U = array.array('d',
                                           [3.35617141647503099647E1,
                                            5.21357949780152679795E2,
                                            4.59432382970980127987E3,
                                            2.26290000613890934246E4,
                                            4.92673942608635921086E4])

cdef array.array Constants_P = array.array('d',
                                           [2.46196981473530512524E-10,
                                            5.64189564831068821977E-1,
                                            7.46321056442269912687E0,
                                            4.86371970985681366614E1,
                                            1.96520832956077098242E2,
                                            5.26445194995477358631E2,
                                            9.34528527171957607540E2,
                                            1.02755188689515710272E3,
                                            5.57535335369399327526E2])

cdef array.array Constants_Q = array.array('d',
                                           [1.32281951154744992508E1,
                                            8.67072140885989742329E1,
                                            3.54937778887819891062E2,
                                            9.75708501743205489753E2,
                                            1.82390916687909736289E3,
                                            2.24633760818710981792E3,
                                            1.65666309194161350182E3,
                                            5.57535340817727675546E2])

cdef array.array Constants_R = array.array('d',
                                           [5.64189583547755073984E-1,
                                            1.27536670759978104416E0,
                                            5.01905042251180477414E0,
                                            6.16021097993053585195E0,
                                            7.40974269950448939160E0,
                                            2.97886665372100240670E0])

cdef array.array Constants_S = array.array('d',
                                           [2.26052863220117276590E0,
                                            9.39603524938001434673E0,
                                            1.20489539808096656605E1,
                                            1.70814450747565897222E1,
                                            9.60896809063285878198E0,
                                            3.36907645100081516050E0])


cdef double[:] cConstants_T = Constants_T
cdef double[:] cConstants_U = Constants_U
cdef double[:] cConstants_P = Constants_P
cdef double[:] cConstants_Q = Constants_Q
cdef double[:] cConstants_R = Constants_R
cdef double[:] cConstants_S = Constants_S


@cython.final
cdef class ClassGaussianEstimator:
    """
    Estimates the distribution of values (according to their class label)
    assuming that they follow the normal distribution
    """
    def __init__(self, int n_classes, int n_bins = 100):
        # settings
        self._n_classes = n_classes
        self._n_bins = n_bins

        # attributes
        self.total = 0
        self._estimators = [GaussianEstimator() for i in range(n_classes)]

        self._max_values = np.empty(n_classes)
        self._max_values[:] = - np.inf
        self.max_value = - np.inf

        self._min_values = np.empty(n_classes)
        self._min_values[:] = np.inf
        self.min_value = np.inf

    cdef double _memory_size(self):
        """
        Returns the amount of RAM memory used in kB
        """
        cdef:
            GaussianEstimator est
            double size

        size = getsizeof(self)
        # compute by hand
        size += getsizeof(self._n_classes)
        size += getsizeof(self._n_bins)
        size += getsizeof(self.total)
        size += getsizeof(self._estimators)
        size += getsizeof(self._max_values)
        size += getsizeof(self._min_values)
        size += getsizeof(self.max_value)
        size += getsizeof(self.min_value)

        size += sum([est.memory_size() for est in self._estimators])
        return size

    cdef _add_instance(self, double value, int y, int weight):
        """
        Updates the respective y GaussianEstimator with a new value
        """
        cdef GaussianEstimator estimator

        if value < self._min_values[y]:
            self._min_values[y] = value
            if value < self.min_value:
                self.min_value = value
        if value > self._max_values[y]:
            self._max_values[y] = value
            if value > self.max_value:
                self.max_value = value
        self.total += weight
        estimator = self._estimators[y]
        estimator.add_value(value, weight)

    cdef np.ndarray[np.double_t, ndim=2] _get_le_gt(self, double split_value):

        """
        Returns the resulting class distributions given a split point
        """
        cdef:
            GaussianEstimator estimator
            int i
            double le, gt
            np.ndarray[np.double_t, ndim=2] le_gt

        le_gt = np.empty((2, self._n_classes), dtype=np.float64)
        for i, estimator in enumerate(self._estimators):
            if split_value < self._min_values[i]:
                le_gt[0, i] = 0
                le_gt[1, i] = estimator._n
            elif split_value >= self._max_values[i]:
                le_gt[0, i] = estimator._n
                le_gt[1, i] = 0
            else:
                le, gt = estimator.estimate_le_gt_distribution(split_value)
                le_gt[0, i] = le
                le_gt[1, i] = gt
        return le_gt

    cdef double _get_proba(self, double value, int y):
        """
        Get the probability of a value belonging to a class y
        """
        cdef:
            GaussianEstimator estimator
            double proba

        estimator = self._estimators[y]
        proba = estimator.proba_density(value)
        # if proba == 0:
        #     return 0
        #     return 1e-10
        return proba

    cdef void _merge(self, ClassGaussianEstimator other):
        """
        Merges two different GaussianNumericEstimators
        """

        for est, other_est in zip(self._estimators, other._estimators):
            est.merge(other_est)

        self._max_values = np.maximum(self._max_values, other._max_values)
        self._min_values = np.minimum(self._min_values, other._min_values)
        self.total += other.total

    def memory_size(self):
        return self._memory_size()

    def add_instance(self, double value, int y, int weight):
        self._add_instance(value, y, weight)

    def get_le_gt(self, double split_value):
        return self._get_le_gt(split_value)

    def merge(self, ClassGaussianEstimator other):
        self._merge(other)

    def get_proba(self, double value, int y):
        return self._get_proba(value, y)

    def possible_splits(self) -> Generator[float]:
        """
        Returns a generator of possible split points
        according to the number of bins setted as parameter
        """

        cdef double step, min_value, max_value

        min_value = self.min_value
        max_value = self.max_value
        if min_value != max_value:
            # new
            step = (max_value - min_value) / (self._n_bins + 1)
            yield from np.linspace(min_value + step, max_value,
                                   self._n_bins + 1, endpoint=False)
            # old
            # yield from np.linspace(min_value, max_value,
            #                        self._n_bins+1)[1:-1]



@cython.final
cdef class GaussianEstimator:
    """
    Used as helper class to the ClassGaussianNumericEstimator
    This class constructs normal distribution from values
    in an online fashion.
    """

    def __init__(self):
        self._n = 0
        self._mean = 0
        self._variance_sum = 0
        self._variance = 0
        self._stdev = 0

    cdef double memory_size(self):
        """
        Returns the used RAM memory in kB
        """

        cdef double size

        size = getsizeof(self)
        # compute by hand
        size += getsizeof(self._n)
        size += getsizeof(self._mean)
        size += getsizeof(self._variance_sum)
        size += getsizeof(self._variance)
        size += getsizeof(self._stdev)

        return size

    cdef add_value(self, double value, int weight):
        """
        Updates the GaussianEstimator with a new value
        """

        cdef double last_mean

        if self._n > 0:
            self._n += weight
            last_mean = self._mean
            self._mean += weight * (value - last_mean) / self._n
            self._variance_sum += (weight * (value - last_mean) *
                                   (value - self._mean))
            self._variance = (self._variance_sum / (self._n - 1)
                              if self._n > 1 else 0)
            self._stdev = sqrt(self._variance)
        else:
            self._mean = value
            self._n = weight

    cpdef merge(self, GaussianEstimator other):
        """
        Merges two different GaussianNumericEstimators
        """

        cdef double delta

        if self._n > 0 and other._n > 0:
            old_mean = self._mean
            new_n = self._n + other._n

            self._mean = ((self._mean * (self._n / new_n)) +
                          (other._mean * (other._n / new_n)))
            self._variance_sum += (other._variance_sum +
                                   (self._n * other._n / new_n) *
                                    pow(other._mean - old_mean, 2))
            self._n = new_n

            self._variance = (self._variance_sum / (self._n - 1)
                              if self._n > 1 else 0)
            self._stdev = sqrt(self._variance)
        elif other._n > 0:
            self._n = other._n
            self._mean = other._mean
            self._variance_sum = other._variance_sum
            self._variance = (self._variance_sum / (self._n - 1)
                              if self._n > 1 else 0)
            self._stdev = sqrt(self._variance)

    cdef double proba_density(self, double value):
        cdef double diff, sq_diff

        if self._n > 0:
            if self._stdev > 0:
                diff = value - self._mean
                sq_diff = diff * diff
                return ((1 / (cConstants_NORMAL_CONST * self._stdev))
                        * exp(-(sq_diff / (2 * self._variance))))
            return 1 if value == self._mean else 0
        return 0

    cdef tuple estimate_le_gt_distribution(self, double split_value):
        """
        Estimates the resulting distribution of values given a split point
        """
        cdef double v, equal_n, less_n, greater_n

        equal_n = self.proba_density(split_value) * self._n
        if self._stdev > 0:
            v = (split_value - self._mean) / self._stdev
            less_n = self.normal_probability(v) * self._n - equal_n
        elif split_value < self._mean:
            less_n = self._n - equal_n
        else:
            less_n = 0
        greater_n = self._n - equal_n - less_n
        if greater_n < 0:
            greater_n = 0
        return less_n + equal_n, greater_n

    cdef double normal_probability(self, double a):
        cdef double x, y, z

        x = a * cConstants_SQRTH
        y = 0.5
        z = abs(x)

        if z < cConstants_SQRTH:
            y += 0.5 * self.error_function(x)
        else:
            y *= self.error_function_complemented(z)
            if x > 0:
                y = 1 - y
        return y

    cdef double error_function(self, double x):
        cdef double z, y

        if abs(x) > 1:
            return 1 - self.error_function_complemented(x)

        z = x * x
        y = (x * self.polevl(z, cConstants_T, 4)
             / self.p1evl(z, cConstants_U, 5))
        return y

    cdef double error_function_complemented(self, double a):
        cdef double x, z, p, q, y

        x = abs(a)

        if x < 1:
            return 1 - self.error_function(a)

        z = -a * a

        if z < - cConstants_MAXLOG:
            if a < 0:
                return 2
            else:
                return 0

        z = exp(z)

        if x < 8:
            p = self.polevl(x, cConstants_P, 8)
            q = self.p1evl(x, cConstants_Q, 8)
        else:
            p = self.polevl(x, cConstants_R, 5)
            q = self.p1evl(x, cConstants_S, 6)

        y = (z * p) / q

        if a < 0:
            y = 2 - y

        if y == 0:
            return 2 if a < 0 else 0.0

        return y

    cdef double polevl(self, double x, double[:] coef, int N):
        cdef:
            int i
            double ans

        ans = coef[0]
        for i in range(1, N + 1):
            ans = ans * x + coef[i]
        return ans

    cdef double p1evl(self, double x, double[:] coef, int N):
        cdef:
            int i
            double ans

        ans = x + coef[0]
        for i in range(1, N):
            ans = ans * x + coef[i]
        return ans
