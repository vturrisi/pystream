# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from sys import getsizeof
from typing import Generator
cimport cython


@cython.final
cdef class NominalCounter:
    """
    A class that performs counting of a nominal attribute
    according to a given class label.
    """

    def __init__(self, int n_classes, tuple values):
        """
        Args:
            n_classes: number of class labels
            values: list of possible values for the nominal attribute
        """

        self._n_classes = n_classes
        self._values = values
        self._counts = {v: np.zeros(n_classes, dtype=int) for v in values}
        self._totals = {v: 0 for v in values}
        self.total = 0

    cdef double _memory_size(self):
        """
        Computes the amount of RAM memory used by the object in bytes
        """

        cdef double size = getsizeof(self)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._values)
        size += getsizeof(self._counts)
        size += getsizeof(self._totals)
        size += getsizeof(self.total)

        size += sum([getsizeof(count) for count in self._counts.values()])
        size += sum([getsizeof(t) for t in self._totals.values()])
        return size

    cdef _add_instance(self, object value, int y, int weight):
        """
        Increments the number of occurences of a value with some weight
        given a class label y

        Args:
            value: the observed value
            y: the class label
            weight: the weight of that given value
        """

        self.total += weight
        self._totals[value] += weight
        self._counts[value][y] += weight

    cdef int _get_count(self, object value):
        """
        Returns the number of occurences of a given value for
        all possible class labels
        """

        return self._counts[value]

    cdef tuple _get_values(self):
        """
        Returns all possible values of that given discrete attribute
        """

        return self._values

    cdef bint _empty(self):
        """
        Returns a boolean indicating if the counter has seen
        at least one instance
        """

        return self.total == 0

    cdef dict _get_distribution(self):
        """
        Returns the distributions of each possible value related
        to the class labels

        Format: {v1: [v1_occ_c1, v1_occ_c2], v2: [v2_occ_c1, v2_occ_c2]}
        """
        return self._counts

    cdef dict _get_total_per_value(self):
        return self._totals

    cdef void _merge(self, NominalCounter other):
        """
        Merges with data from a different nominal counters
        (for the same discrete variable)

        Args:
            other: Other NominalCounter to merge with
        """

        cdef int v
        self.total = other.total
        for v in self._counts:
            self._counts[v] += other._counts[v]

    cdef double _get_proba(self, object value, int y):
        """
        Get the probability of a value belonging to a specific class

        Args:
            value:
        """

        cdef double occ, total

        occ = self._counts[value][y]
        total = self._totals[value]
        if occ == 0:
            return 0
            # return 1e-10
        return occ / total

    cdef dict _get_probas(self):
        cdef:
            double n, tot
            np.ndarray dist
            dict probas

        probs = {}
        for v, dist in self._counts.items():
            tot = self._totals[v]
            if tot != 0:
                probs[v] = dist / tot
            else:
                probs[v] = np.zeros(self._n_classes)
        return probs

    cdef tuple _possible_values(self):
        """
        Returns a generator of possible split points
        according to the number of bins setted as parameter
        """

        return self._values

    def memory_size(self):
        return self._memory_size()

    def add_instance(self, object value, int y, int weight):
        self._add_instance(value, y, weight)

    def get_count(self, object value):
        return self._get_count(value)

    def get_values(self):
        return self._get_values()

    def empty(self):
        return self._empty()

    def get_distribution(self):
        return self._get_distribution()

    def get_total_per_value(self):
        return self._get_total_per_value()

    def merge(self, NominalCounter other):
        self._merge(other)

    def get_proba(self, object value, int y):
        return self._get_proba(value, y)

    def get_probas(self):
        return self._get_probas()

    def possible_values(self):
        return self._possible_values()
