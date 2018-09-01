# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY

from sys import getsizeof
from ..change_detectors.adwin import ADWIN
from ..change_detectors.adwin cimport ADWIN


cdef class LeveragingBagging:
    cdef object _attr_types
    cdef int _n_classes, _n_predictors
    cdef object _base_learner
    cdef str _base_learner_name
    cdef double _adwin_delta
    cdef dict _base_learner_kwargs, _learners, _learners_n, _adwins

    def __init__(self, attr_types, n_classes, n_predictors, *,
                 base_learner, adwin_delta=0.0001,
                 base_learner_kwargs={}):
        self._attr_types = attr_types
        self._n_classes = n_classes
        self._n_predictors = n_predictors

        # adwin
        self._adwins = {i: ADWIN(adwin_delta)
                        for i in range(self._n_predictors)}
        self._adwin_delta = adwin_delta

        # base learners
        self._base_learner = base_learner
        self._base_learner_kwargs = base_learner_kwargs
        self._base_learner_name = base_learner.__name__
        self._learners = {i: base_learner(attr_types, n_classes,
                                          **base_learner_kwargs)
                          for i in range(self._n_predictors)}
        self._learners_n = {i: 0 for i in range(self._n_predictors)}

    cdef list _get_stats(self):
        return [learner.get_stats() for learner in self._learners.values()]

    cdef double _memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._attr_types)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._n_predictors)
        size += getsizeof(self._base_learner)
        size += getsizeof(self._base_learner_kwargs)
        size += getsizeof(self._base_learner_name)
        size += getsizeof(self._learners)
        size += getsizeof(self._adwin_delta)

        # learners memory
        size += sum([learner.memory_size()
                    for learner in self._learners.values()])
        size += sum([getsizeof(n) for n in self._learners_n.values()])

        size += sum([adwin.memory_size() for adwin in self._adwins.values()])

        return size

    cdef tuple _get_highest_error(self):
        cdef:
            int i
            double error

        error, i = max([(self._adwins[i].get_estimation(), i)
                        for i in self._learners],
                       key=lambda error_i: error_i[0])
        return error, i

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef int pred, i
        cdef object learner
        cdef bint drifted
        cdef double error, error_est

        drifted = False
        # train all classifiers
        for i, learner in self._learners.items():
            learner.train(X, y, weight * np.random.poisson(6))
            self._learners_n[i] += weight

            # update adwin and check for drifts
            pred = learner.predict(X)
            if pred != y:
                error = 1
            else:
                error = 0
            error_est = self._adwins[i].get_estimation()
            # if there was a drift and the error increased
            if (self._adwins[i].update(error) and
                self._adwins[i].get_estimation() > error_est):
                    drifted = True

        # if adwin detects a drift in any classifier, drop worst
        # performing one
        if drifted:
            error, i = self._get_highest_error()
            self._learners[i] = self._base_learner(self._attr_types,
                                                   self._n_classes,
                                                   **self._base_learner_kwargs)
            self._adwins[i] = ADWIN(self._adwin_delta)

    cdef int _predict(self, np.ndarray X):
        cdef int yhat = -1, i
        cdef object learner
        cdef np.ndarray predictions
        cdef double v = -INFINITY, p

        predictions = np.zeros(self._n_classes)
        for i, learner in self._learners.items():
            pred = learner.predict(X)
            predictions[pred] += 1

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
