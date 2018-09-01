# cython: boundscheck=False, wraparound=False

from sys import getsizeof


cdef class TreeStats:
    def __init__(self, int n_classes):
        self._n_classes = n_classes
        self._stats = {'splits': 0, 'splits_blocked': 0,
                       'n_nodes': 1, 'n_leaves': 1}

    def __getitem__(self, str key):
        return self._stats[key]

    def __setitem__(self, str key, int value):
        self._stats[key] = value

    cdef double _memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._stats)
        size += sum([getsizeof(value) for value in self._stats.values()])
        return size

    def memory_size(self):
        return self._memory_size()
