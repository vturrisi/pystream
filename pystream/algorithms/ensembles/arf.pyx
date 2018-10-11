# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY

from sys import getsizeof
from ..change_detectors.adwin import ADWIN


cdef class AdaptiveRandomForests:
    cdef object _attr_types
    cdef int _n_classes, _n_predictors
    cdef double _warning_delta, _drift_delta
    cdef int _m_attrs
    cdef object _base_learner
    cdef str _base_learner_name
    cdef dict _base_learner_kwargs, _learners, _learners_n, _learners_weight, \
        _warning_adwins, _drift_adwins, _bkgr_learners, _bkgr_learners_n, \
        _bkgr_learners_weight

    def __init__(self, attr_types, n_classes, n_predictors,
                 warning_delta, drift_delta, *,
                 base_learner, base_learner_kwargs={}, m_attrs=None):
        self._attr_types = attr_types
        self._n_classes = n_classes
        self._n_predictors = n_predictors
        self._warning_delta = warning_delta
        self._drift_delta = drift_delta

        if m_attrs is None:
            n_attrs = len(attr_types)
            self._m_attrs = round(n_attrs ** (1 / 2) + 1)
        else:
            self._m_attrs = m_attrs

        # base learners
        self._base_learner = base_learner
        self._base_learner_kwargs = base_learner_kwargs
        self._base_learner_name = base_learner.__name__
        # start with one base learner
        self._learners = {i: base_learner(attr_types, n_classes,
                                          m_attrs=self._m_attrs,
                                          **base_learner_kwargs)
                          for i in range(self._n_predictors)}
        self._learners_n = {i: 0 for i in range(self._n_predictors)}
        self._learners_weight = {i: 1 for i in range(self._n_predictors)}
        self._warning_adwins = {i: ADWIN(self._warning_delta)
                                for i in range(self._n_predictors)}
        self._drift_adwins = {i: ADWIN(self._drift_delta)
                              for i in range(self._n_predictors)}

        self._bkgr_learners = {}
        self._bkgr_learners_n = {}
        self._bkgr_learners_weight = {}

    cdef list _get_stats(self):
        return [learner.get_stats() for learner in self._learners.values()]

    cdef double _memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._attr_types)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._n_predictors)
        size += getsizeof(self._warning_delta)
        size += getsizeof(self._drift_delta)
        size += getsizeof(self._m_attrs)
        size += getsizeof(self._base_learner)
        size += getsizeof(self._base_learner_kwargs)
        size += getsizeof(self._base_learner_name)
        size += getsizeof(self._learners)
        size += getsizeof(self._learners_n)
        size += getsizeof(self._learners_weight)
        size += getsizeof(self._warning_adwins)
        size += getsizeof(self._drift_adwins)
        size += getsizeof(self._bkgr_learners)
        size += getsizeof(self._bkgr_learners_n)
        size += getsizeof(self._bkgr_learners_weight)

        # learners memory
        size += sum([learner.memory_size()
                    for learner in self._learners.values()])
        # learners memory
        size += sum([getsizeof(weight)
                     for weight in self._learners_weight.values()])
        size += sum([getsizeof(n)
                     for n in self._learners_n.values()])

        # background learners memory
        size += sum([learner.memory_size()
                     for learner in self._bkgr_learners.values()])
        size += sum([getsizeof(weight)
                     for weight in self._bkgr_learners_weight.values()])
        size += sum([getsizeof(n)
                     for n in self._bkgr_learners_n.values()])

        size += sum([adwin.memory_size()
                     for adwin in self._warning_adwins.values()])
        size += sum([adwin.memory_size()
                     for adwin in self._drift_adwins.values()])

        return size

    cdef void _update_weight(self, int correctly_classifies, int i):
        cdef double weight, n

        weight = self._learners_weight[i]
        n = self._learners_n[i]
        self._learners_weight[i] = ((weight * n + correctly_classifies)
                                    / (n + 1))

    cdef void _init_bkg_learner(self, int i):
        new_learner = self._base_learner(self._attr_types,
                                         self._n_classes,
                                         m_attrs=self._m_attrs,
                                         **self._base_learner_kwargs)
        self._bkgr_learners[i] = new_learner
        self._bkgr_learners_n[i] = 0
        self._bkgr_learners_weight[i] = 1
        self._warning_adwins[i] = ADWIN(self._warning_delta)

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef int i, correctly_classifies, yhat
        cdef double error, error_est
        cdef object learner

        for i, learner in self._learners.items():
            # update weight
            correctly_classifies = 1
            error = 0
            if self._learners_n[i] > 0:
                yhat = learner.predict(X)
                if y == yhat:
                    correctly_classifies = 1
                    error = 0
                else:
                    correctly_classifies = 0
                    error = 1
                self._update_weight(correctly_classifies, i)

            # learn on instance
            learner.train(X, y, weight=weight * np.random.poisson(6))
            self._learners_n[i] += 1

            # concept drift warning -> create new tree
            error_est = self._warning_adwins[i].get_estimation()
            # if there was a drift and the error increased
            if (self._warning_adwins[i].update(error) and
                self._warning_adwins[i].get_estimation() > error_est):
                self._init_bkg_learner(i)

            # detected concept drift -> replace current tree with bkgr tree
            error_est = self._drift_adwins[i].get_estimation()
            # if there was a drift and the error increased
            if (self._drift_adwins[i].update(error) and
               self._drift_adwins[i].get_estimation() > error_est):
                if i not in self._bkgr_learners:
                    self._init_bkg_learner(i)
                self._learners[i] = self._bkgr_learners[i]
                self._learners_n[i] = self._bkgr_learners_n[i]
                self._learners_weight[i] = self._bkgr_learners_weight[i]
                self._drift_adwins[i] = ADWIN(self._drift_delta)

                del self._bkgr_learners[i]
                del self._bkgr_learners_n[i]
                del self._bkgr_learners_weight[i]

        for i, learner in self._bkgr_learners.items():
            learner.train(X, y, weight=weight * np.random.poisson(6))
            self._bkgr_learners_n[i] += 1

    cdef int _predict(self, np.ndarray X):
        cdef int i, yhat = -1
        cdef np.ndarray predictions
        cdef double v = -INFINITY, p
        cdef object learner

        predictions = np.zeros(self._n_classes)
        for i, learner in self._learners.items():
            pred = learner.predict(X)
            predictions[pred] += self._learners_weight[i]

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
