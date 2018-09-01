# cython: boundscheck=False, wraparound=False

from libc.math cimport sqrt
from sys import getsizeof


cdef class SimpleStats:
    def __init__(self):
        self._stats = {'n': 0, 'mean': 0, 'var_sum': 0, 'var': 0, 'stdev': 0}

    def __getitem__(self, str key):
        return self._stats[key]

    cdef double _memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._stats)
        size += sum([getsizeof(value) for value in self._stats.values()])
        return size

    cdef void _update(self, double value):
        cdef int n
        cdef double last_mean, mean, var_sum, var

        self._stats['n'] += 1
        n = self._stats['n']

        last_mean = self._stats['mean']
        self._stats['mean'] += (value - last_mean) / n
        mean = self._stats['mean']

        self._stats['var_sum'] += (value - last_mean) * (value - mean)
        var_sum = self._stats['var_sum']

        self._stats['var'] = (var_sum / (n - 1)) if n > 1 else 0
        var = self._stats['var']

        self._stats['stdev'] = sqrt(var)

    cdef void _remove(self, double value):
        assert self._stats['n'] > 1

        cdef int n
        cdef double last_mean, mean, var_sum, var

        self._stats['n'] -= 1
        n = self._stats['n']

        last_mean = self._stats['mean']
        self._stats['mean'] -= (value - last_mean) / n
        mean = self._stats['mean']

        self._stats['var_sum'] -= (value - last_mean) * (value - mean)
        var_sum = self._stats['var_sum']

        self._stats['var'] = (var_sum / (n - 1)) if n > 1 else 0
        var = self._stats['var']

        self._stats['stdev'] = sqrt(var)

    def memory_size(self):
        return self._memory_size()

    def update(self, double value):
        self._update(value)

    def remove(self, double value):
        self._remove(value)

    @property
    def mean(self):
        return self._stats['mean']

    @property
    def var(self):
        return self._stats['var']

    @property
    def std(self):
        return self._stats['stdev']

    @property
    def n(self):
        return self._stats['n']


cdef class NormalDistValueStats(SimpleStats):
    def __init__(self):
        super().__init__()

    cdef double _mean_plus_factor_std(self, double factor):
        return self._stats['mean'] + factor * self._stats['stdev']

    def mean_plus_factor_std(self, double factor):
        return self._mean_plus_factor_std(factor)

    def mean_plus_std(self):
        return self._mean_plus_factor_std(1)

    def mean_minus_std(self):
        return self._mean_plus_factor_std(-1)
