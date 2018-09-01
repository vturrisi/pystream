# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from sys import getsizeof
from libc.math cimport log, INFINITY


cdef class OzaBoosting:
    cdef object _attr_types
    cdef int _n_classes, _n_predictors
    cdef object _base_learner
    cdef dict _base_learner_kwargs
    cdef str _base_learner_name
    cdef list _learners
    cdef np.ndarray _learners_correct, _learners_wrong

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
        self._learners_correct = np.zeros(n_predictors)
        self._learners_wrong = np.zeros(n_predictors)

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
        size += getsizeof(self._learners_correct)
        size += getsizeof(self._learners_wrong)
        # learners memory
        size += sum([learner.memory_size()
                     for learner in self._learners])
        return size

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef int yhat, l_num
        cdef object base_learner
        cdef double N, w
        w = weight
        # boosting
        for l_num, base_learner in enumerate(self._learners):
            if w == 0:
                break
            base_learner.train(X, y, weight=np.random.poisson(w))
            yhat = base_learner.predict(X)
            if yhat == y:
                self._learners_correct[l_num] += w
                N = self._learners_correct[l_num] + self._learners_wrong[l_num]
                w *= N / (2 * self._learners_correct[l_num])
            else:
                self._learners_wrong[l_num] += w
                N = self._learners_correct[l_num] + self._learners_wrong[l_num]
                w *= N / (2 * self._learners_wrong[l_num])

    cdef int _predict(self, np.ndarray X):
        cdef np.ndarray predictions = np.zeros(self._n_classes)
        cdef int pred, l_num, yhat = -1
        cdef object learner
        cdef double e, weight, b, wrong, correct
        cdef double v = -INFINITY, p

        for l_num, learner in enumerate(self._learners):
            pred = learner.predict(X)
            wrong = self._learners_wrong[l_num]
            correct = self._learners_correct[l_num]
            if wrong == 0:
                e = 0
            else:
                e = wrong / (correct + wrong)

            if e == 0 or e > 0.5:
                weight = 0
            else:
                b = e / (1 - e)
                weight = log(1 / b)
            predictions[pred] += weight

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
