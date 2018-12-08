# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY
from sys import getsizeof


cdef class OzaBagging:
    cdef object _attr_types
    cdef int _n_classes, _n_predictors
    cdef object _base_learner
    cdef dict _base_learner_kwargs
    cdef str _base_learner_name
    cdef list _learners

    def __init__(self, attr_types, n_classes, n_predictors, *,
                 base_learner, base_learner_kwargs={}):
        self._attr_types = attr_types
        self._n_classes = n_classes
        self._n_predictors = n_predictors

        # base learners
        self._base_learner = base_learner
        self._base_learner_kwargs = base_learner_kwargs
        self._base_learner_name = base_learner.__name__
        self._learners = [base_learner(attr_types, n_classes,
                                       **base_learner_kwargs)
                          for i in range(n_predictors)]

    cdef list _get_stats(self):
        return [learner.get_stats() for learner in self._learners]

    cdef double _memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._attr_types)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._n_predictors)
        size += getsizeof(self._base_learner)
        size += getsizeof(self._base_learner_kwargs)
        size += getsizeof(self._base_learner_name)
        size += getsizeof(self._learners)

        # learners memory
        size += sum([learner.memory_size()
                     for learner in self._learners])
        return size

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef object learner
        cdef int k

        for learner in self._learners:
            k = weight * np.random.poisson(1)
            if k > 0:
                learner.train(X, y, k)

    cdef int _predict(self, np.ndarray X):
        cdef object learner
        cdef int yhat = -1
        cdef np.ndarray predictions
        cdef double v = -INFINITY, p

        predictions = np.zeros(self._n_classes)
        for learner in self._learners:
            predictions[learner.predict(X)] += 1

        for i in range(predictions.shape[0]):
            p = predictions[i]
            if p > v:
                v = p
                yhat = i
        return yhat
        # yhat = np.argmax(predictions)
        # return yhat

    def get_stats(self):
        return self._get_stats()

    def memory_size(self):
        return self._memory_size()

    def train(self, np.ndarray X, int y, int weight=1):
        return self._train(X, y, weight)

    def predict(self, np.ndarray X):
        return self._predict(X)
