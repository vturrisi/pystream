# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY

from sys import getsizeof


cdef class OnlineAccuracyUpdatedEnsemble:
    cdef object _attr_types
    cdef int _n_classes, _n_predictors, _window_size, _t, _candidate_t
    cdef object _base_learner, _candidate_learner
    cdef str _base_learner_name
    cdef double _candidate_weight, _MSEr, _memory_limit
    cdef dict _base_learner_kwargs, _learners, _learners_t, \
        _learners_weight, _MSEi, _errors
    cdef np.ndarray _class_dist, _current_window

    def __init__(self, attr_types, n_classes, n_predictors, window_size, *,
                 base_learner, base_learner_kwargs={}, memory_limit=-1):
        self._attr_types = attr_types
        self._n_classes = n_classes
        self._n_predictors = n_predictors
        self._window_size = window_size
        self._memory_limit = memory_limit

        self._t = 0
        # base learners
        self._base_learner = base_learner
        self._base_learner_kwargs = base_learner_kwargs
        self._base_learner_name = base_learner.__name__
        # start all base learners
        self._learners = {}
        self._learners_t = {}
        self._learners_weight = {}

        # save errors in last iteration
        self._MSEi = {}
        self._MSEr = 0
        self._errors = {}
        self._current_window = np.negative(np.ones(self._window_size))
        self._class_dist = np.zeros(self._n_classes, dtype=int)

        self._candidate_learner = \
            self._base_learner(self._attr_types, self._n_classes,
                               **self._base_learner_kwargs)

    cdef list _get_stats(self):
        return [learner.get_stats() for learner in self._learners.values()]

    cdef double _memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._attr_types)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._n_predictors)
        size += getsizeof(self._window_size)
        size += getsizeof(self._memory_limit)
        size += getsizeof(self._t)
        size += getsizeof(self._base_learner)
        size += getsizeof(self._base_learner_kwargs)
        size += getsizeof(self._base_learner_name)
        size += getsizeof(self._learners)
        size += getsizeof(self._learners_t)
        size += getsizeof(self._learners_weight)
        size += getsizeof(self._MSEi)
        size += getsizeof(self._MSEr)
        size += getsizeof(self._errors)
        size += getsizeof(self._current_window)
        size += getsizeof(self._class_dist)
        size += getsizeof(self._candidate_t)

        size += self._candidate_learner.memory_size()

        # learners memory
        size += sum([learner.memory_size()
                    for learner in self._learners.values()])
        size += sum([getsizeof(t) for t in self._learners_t.values()])
        size += sum([getsizeof(w) for w in self._learners_weight.values()])
        size += sum([getsizeof(mse) for mse in self._MSEi.values()])
        size += sum([getsizeof(e) for e in self._errors.values()])

        return size

    cdef void _add_classifier_to_ensemble(self):
        cdef int id_

        if len(self._learners) == 0:
            id_ = 0
        else:
            id_ = max(self._learners.keys()) + 1
        self._learners[id_] = self._candidate_learner
        self._learners_t[id_] = self._candidate_t
        self._learners_weight[id_] = self._candidate_weight
        self._errors[id_] = np.negative(np.ones(self._window_size))
        self._MSEi[id_] = 0

    cdef tuple _get_worst_classifier(self):
        cdef int id_
        cdef double weight

        id_, weight = min(self._learners_weight.items(),
                          key=lambda i_weight: i_weight[1])
        return id_, weight

    cdef void _remove_classifier(self, int id_):
        del self._learners[id_]
        del self._learners_t[id_]
        del self._learners_weight[id_]
        del self._MSEi[id_]
        del self._errors[id_]

    cdef void _create_new_candidate(self, np.ndarray X, int y, int t):
        cdef double _candidate_weight, weight
        cdef int worst_id

        self._candidate_weight = 1 / (self._MSEr + 10e-300)
        self._candidate_t = t
        if len(self._learners) < self._n_predictors:
            self._add_classifier_to_ensemble()
        else:
            worst_id, weight = self._get_worst_classifier()
            if weight < self._candidate_weight:
                self._remove_classifier(worst_id)
            self._add_classifier_to_ensemble()

        self._candidate_learner = \
            self._base_learner(self._attr_types, self._n_classes,
                               **self._base_learner_kwargs)

    cdef void _compute_weights(self, np.ndarray X, int y):
        cdef double e, total, mse_i, p
        cdef np.ndarray probs
        cdef int d, t, id_
        cdef object classifier

        d = self._window_size
        for id_, classifier in self._learners.items():
            t = self._t - self._learners_t[id_]

            e = 0
            probs = classifier.predict_proba(X)
            total = 0
            for p in probs:
                total += p
            # total = np.sum(probs)
            if total > 0:
                e = (1 - probs[y] / total) ** 2

            if t > d:
                mse_i = self._MSEi[id_] + e / d - self._errors[id_][t % d] / d
            else:
                mse_i = self._MSEi[id_] * (t - 1) / t + e / d

            self._MSEi[id_] = mse_i
            self._errors[id_][t % d] = e
            self._learners_weight[id_] = 1 / (mse_i + self._MSEr + 10e-10)

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef double to_decrement
        # increment instance classes seen in window and
        # class distribution of the current window
        if self._t >= self._window_size:
            to_decrement = self._current_window[self._t % self._window_size]
            self._class_dist[int(to_decrement)] -= 1

        self._class_dist[y] += 1
        self._current_window[self._t % self._window_size] = y

        # compute errors
        self._t += 1
        self._compute_MSEr()

        # update weights for the ensemble
        self._compute_weights(X, y)

        if self._t > 0 and self._t % self._window_size == 0:
            self._create_new_candidate(X, y, self._t)
        else:
            # train candidate classifier on instance
            self._candidate_learner.train(X, y, weight)

        # train ensemble on instance
        for learner in self._learners.values():
            learner.train(X, y, weight)

    cdef void _compute_MSEr(self):
        cdef double count, prob

        self._MSEr = 0
        for count in self._class_dist:
            prob = count / self._window_size
            self._MSEr += prob * ((1 - prob) * (1 - prob))

    cdef dict _learners_predict(self, np.ndarray X):
        cdef dict predictions
        cdef int i
        cdef object learner
        cdef np.ndarray probs

        predictions = {}
        for i, learner in self._learners.items():
            probs = learner.predict_proba(X)
            predictions[i] = probs
        return predictions

    cdef int _predict(self, np.ndarray X):
        cdef double max_weight, min_weight, weight
        cdef int i, pred, yhat = -1
        cdef np.ndarray predictions
        cdef object learner
        cdef double v = -INFINITY, p

        if len(self._learners) == 0:
            return self._candidate_learner.predict(X)
        max_weight = max(self._learners_weight.values())
        min_weight = min(self._learners_weight.values())
        predictions = np.zeros(self._n_classes)
        for i, learner in self._learners.items():
            pred = learner.predict(X)
            if max_weight == min_weight:
                weight = 1
            else:
                weight = ((self._learners_weight[i] - min_weight)
                          / (max_weight - min_weight))
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
