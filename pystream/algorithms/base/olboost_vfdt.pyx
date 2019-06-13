# cython: boundscheck=False, wraparound=False

cimport numpy as np
from vfdt cimport VFDT, Node, PossibleSplit
from cpython cimport PyObject_DelAttr

import numpy as np
from sys import getsizeof
from libc.math cimport log, log2, pow, INFINITY
from numeric_estimators import ClassGaussianEstimator
from nominal_counters import NominalCounter
from vfdt import VFDT, Node, Distribution, Iterable, Optional


cdef class BoostNode(Node):
    def __init__(self, name: int, attr_types: Iterable, int n_classes,
                 object numeric_estimator, str prediction_type,
                 bint only_binary_splits, set exhausted_attrs,
                 object m_attrs=None, BoostNode parent=None,
                 distribution: Distribution = None):

        Node.__init__(self, name=name, attr_types=attr_types,
                      n_classes=n_classes,
                      numeric_estimator=numeric_estimator,
                      prediction_type=prediction_type,
                      only_binary_splits=only_binary_splits,
                      exhausted_attrs=exhausted_attrs,
                      m_attrs=m_attrs, parent=parent,
                      distribution=distribution)

        # if this node in inheriting some
        # class distribution from its parent
        self._dist_pred = np.copy(self._dist)

        if prediction_type in ('adaptive', 'nb'):
            self._nb_estimators = [numeric_estimator(self._n_classes)
                                   if t is float else
                                   NominalCounter(self._n_classes, values=t)
                                   for t in self._attr_types]

    cdef double _memory_size(self):
        cdef double size

        size = Node._memory_size(self)
        if (self._split_type is None and
            self._prediction_type in ('adaptive', 'nb')):
            size += sum([estimator.memory_size()
                         for estimator in self._nb_estimators])

        return size

    cdef void clean_memory(self):
        Node.clean_memory(self)
        PyObject_DelAttr(self, '_dist_pred')
        if self._prediction_type in ('adaptive', 'nb'):
            PyObject_DelAttr(self, '_nb_estimators')

    cdef void _update_nb_estimators(self, np.ndarray X, int y, int weight):
        # flag for naive bayes
        self._estimators_empty = False

        for estimator, x in zip(self._nb_estimators, X):
            estimator.add_instance(x, y, weight)

    cdef double _compute_lambda_weight(self, double prob,
                                       double min_l, double max_l):
        return (min_l - max_l) * prob + max_l


    cdef void learn_from_instance_boost(self, np.ndarray X, int y, int weight,
                                        int min_l, int max_l):
        cdef:
            double lamb
            np.ndarray probs

        # updates acc
        # if self._prediction_type == 'mc':
        #     probs, yhat = self._mc_update_accs(X, y, weight)
        #     if yhat != y:
        #         lamb = max_l
        #     else:
        #         lamb = min_l
        #     prediction_weight = weight * np.random.poisson(lamb)
        # else:
        #     if self._prediction_type == 'adaptive':
        #         probs, yhat = self._adaptive_update_accs(X, y, weight)
        #     elif self._prediction_type == 'nb':
        #         probs, yhat = self._nb_update_accs(X, y, weight)
        #     if yhat != y:
        #         lamb = max_l
        #     else:
        #         lamb = min_l
        #     prediction_weight = weight * np.random.poisson(lamb)
        #     self._update_nb_estimators(X, y, prediction_weight)
        if self._prediction_type == 'mc':
            probs, _ = self._mc_update_accs(X, y, weight)
            lamb = self._compute_lambda_weight(probs[y], min_l, max_l)
            prediction_weight = weight * np.random.poisson(lamb)
        else:
            if self._prediction_type == 'adaptive':
                probs, _ = self._adaptive_update_accs(X, y, weight)
            elif self._prediction_type == 'nb':
                probs, _ = self._nb_update_accs(X, y, weight)
            lamb = self._compute_lambda_weight(probs[y], min_l, max_l)
            prediction_weight = weight * np.random.poisson(lamb)
            self._update_nb_estimators(X, y, prediction_weight)

        # for prediction
        self._dist_pred[y] += prediction_weight

        # for splitting
        self._update_attr_estimators(X, y, weight)
        self._dist[y] += + weight

        # increase the number of elements seen
        self._n += weight

    cdef np.ndarray mc_predict_proba(self):
        cdef:
            np.ndarray probs
            double tot, p

        if self._n == 0:
            probs = np.empty(self._n_classes)
            probs[:] = 1.0 / self._n_classes
            return probs
        tot = 0
        for p in self._dist_pred:
            tot += p
        if tot != 0:
            return self._dist_pred / tot
        else:
            return self._dist

    cdef int mc_predict(self):
        if self._n == 0:
            return np.random.randint(0, self._n_classes)

        cdef int yhat = -1, i
        cdef double v = -INFINITY, p
        cdef np.ndarray probs

        probs = self._dist_pred
        for i in range(self._dist_pred.shape[0]):
            p = self._dist_pred[i]
            if p > v:
                v = p
                yhat = i
        return yhat
        # return np.argmax(self._dist)

    cdef np.ndarray nb_predict_proba(self, np.ndarray X):
        cdef:
            object x, estimator
            np.ndarray probs
            int y, i
            double tot

        probs = self.mc_predict_proba()
        if not self._estimators_empty:
            for y in range(self._n_classes):
                for x, estimator in zip(X, self._nb_estimators):
                    probs[y] *= estimator.get_proba(x, y)

            # scale prob output
            tot = 0
            for p in probs:
                tot += p
            if tot != 0:
                for i in range(probs.shape[0]):
                    probs[i] /= tot
                return probs
        return probs

    cdef np.ndarray nb_predict_log_proba(self, np.ndarray X):
        cdef:
            object x, estimator
            np.ndarray probs
            int y, i
            double tot

        probs = np.log(self.mc_predict_proba())
        if not self._estimators_empty:
            for y in range(self._n_classes):
                for x, estimator in zip(X, self._nb_estimators):
                    probs[y] += log(estimator.get_proba(x, y))

            # scale prob output
            tot = 0
            for p in probs:
                tot += p
            if tot != 0:
                for i in range(probs.shape[0]):
                    probs[i] /= tot
                return probs
        return probs


cdef class OLBoostVFDT(VFDT):
    def __init__(self, attr_types: Iterable, int n_classes, *,
                 int gp=200,
                 double delta=1e-5, double tiebreaker=0.05,
                 int min_l=1, int max_l=6,
                 object numeric_estimator=ClassGaussianEstimator,
                 str prediction_type='adaptive',
                 str split_criterion='infogain',
                 bint only_binary_splits=False,
                 bint drop_poor_attrs=True,
                 bint clean_after_split=True,
                 m_attrs: Optional[int]=None,
                 object node_class=BoostNode):

        VFDT.__init__(self,
                      attr_types=attr_types,
                      n_classes=n_classes,
                      gp=gp,
                      delta=delta,
                      tiebreaker=tiebreaker,
                      numeric_estimator=numeric_estimator,
                      prediction_type=prediction_type,
                      split_criterion=split_criterion,
                      only_binary_splits=only_binary_splits,
                      drop_poor_attrs=drop_poor_attrs,
                      clean_after_split=clean_after_split,
                      m_attrs=m_attrs,
                      node_class=node_class)

        self._min_l = min_l
        self._max_l = max_l

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef:
            BoostNode leaf
            double gp, current_n, last_n, hb
            list rank
            PossibleSplit split

        if weight == 0:
            return

        leaf = self._sort_to_leaf(X)

        leaf.learn_from_instance_boost(X, y, weight, self._min_l, self._max_l)

        # all attributes were exhausted for that leaf
        # the chances of this happening are very low and all attributes
        # need to be nominal, but this may save some computational time
        # (and is better than setting leaf._last_n to infinity)
        if leaf.active:
            gp = self._grace_period
            current_n = leaf._n
            last_n = leaf._last_n

            if current_n - last_n >= gp and not leaf.all_same_class():
                leaf._last_n = current_n
                # compute the gain of all attributes and sort in descending way
                rank_func = getattr(leaf, self._rank_function)
                rank = rank_func()

                # this means that all attributes were exhausted_attrs
                # and we will never be able to split this leaf any further
                if len(rank) == 0:
                    leaf.active = False
                else:
                    hb = self._hoeffding_bound(delta=self._delta,
                                               R=log2(self._n_classes),
                                               n=current_n)
                    if self._can_split(rank, hb, leaf):
                        split = rank[0]
                        self._split_leaf(split, leaf)
                        # Just delete the estimators to free up some memory
                        if self._clean_after_split:
                            leaf.clean_memory()
                    elif self._drop_poor_attrs:
                        leaf._drop_poor_attrs_func(rank, hb)
