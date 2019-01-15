# cython: boundscheck=False, wraparound=False

cimport numpy as np
cimport cython
from cpython cimport PyObject_DelAttr
from libc.math cimport sqrt, log, log2, isfinite, INFINITY

import numpy as np
import weakref
from sys import getsizeof
from random import sample
from typing import Iterable, Optional
from collections import namedtuple
from numeric_estimators import ClassGaussianEstimator
from nominal_counters import NominalCounter
from statistics.tree_stats import TreeStats
from statistics.tree_stats cimport TreeStats


# type definitions
Distribution = Iterable[float]


cdef class PossibleSplit:
    def __init__(self, attr, attr_type, value, le_set, gain, distribution):
        self.attr = attr
        self.attr_type = attr_type
        self.value = value
        self.le_set = le_set
        self.gain = gain
        self.distribution = distribution

    def __repr__(self):
        return (f'PossibleSplit(attr={self.attr}, '
                f'attr_type={self.attr_type}, '
                f'value={self.value}, le_set={self.le_set}, '
                f'gain={self.gain}, distribution={self.distribution})')


cdef class Node:
    def __init__(self, name: int, attr_types: Iterable, int n_classes,
                 object numeric_estimator, str prediction_type,
                 bint only_binary_splits, set exhausted_attrs,
                 object m_attrs=None, Node parent=None,
                 distribution: Distribution = None):

        self.active = True
        self._name = name
        self._attr_types = attr_types
        self._n_attrs = len(attr_types)
        self._n_classes = n_classes
        self._numeric_estimator = numeric_estimator
        self._prediction_type = prediction_type
        self._only_binary_splits = only_binary_splits
        self._m_attrs = m_attrs

        if parent:
            self._parent = weakref.ref(parent)
        else:
            self._parent = None

        # if there is some split using a nominal attr
        if exhausted_attrs:
            self._exhausted_attrs = set(exhausted_attrs)
            self._dropped_attrs = set(exhausted_attrs)
        else:
            self._exhausted_attrs = set()
            self._dropped_attrs = set()

        if m_attrs is not None:
            # count number of candidate attributes
            attr_being_evaluated = self._n_attrs - len(self._dropped_attrs)
            # select m attributes (if there is less than m do nothing)
            n_remove = attr_being_evaluated - m_attrs
            if n_remove > 0:
                attrs = list(filter(
                             lambda attr: attr not in self._dropped_attrs,
                             range(self._n_attrs)))
                remove_from_comp = set(sample(attrs, n_remove))
                self._dropped_attrs |= remove_from_comp

        # if this node in inheriting some
        # class distribution from its parent
        if distribution is None:
            self._dist_from_split = None
            self._dist = np.zeros(n_classes)
            self._n_from_split = 0
            self._n = 0
            self._last_n = 0
        else:
            self._dist_from_split = np.copy(distribution)
            self._dist = np.copy(distribution)
            self._n_from_split = distribution.sum()
            self._n = self._n_from_split
            self._last_n = self._n

        # initially this node is a leaf
        self._is_leaf = True
        self._children = {}
        self._split_attr = None
        self._split_value = None
        self._split_gain = None
        self._split_type = None
        self._le_set = None

        # initiate attr estimators
        self._estimators_empty = True
        self._attr_estimators = [numeric_estimator(self._n_classes)
                                 if t is float
                                 else NominalCounter(self._n_classes, values=t)
                                 for t in self._attr_types]

        self._stats = {'missed_mc': 0, 'missed_nb': 0,
                       'elements_seen': 0, 'acc': 0}

    cdef void _reset(self):
        '''
        Resets this node, i.e., instantiate new estimators/distributions
        '''

        self.__init__(name=self._name, attr_types=self._attr_types,
                      n_classes=self._n_classes,
                      numeric_estimator=self._numeric_estimator,
                      prediction_type=self._prediction_type,
                      exhausted_attrs=self._exhausted_attrs,
                      m_attrs=self._m_attrs, parent=self.parent)

    cdef double _memory_size(self):
        """
        Computes the amount of RAM memory used by the object in bytes
        """

        cdef double size

        size = getsizeof(self)
        size += getsizeof(self.active)
        size += getsizeof(self._is_leaf)
        size += getsizeof(self._only_binary_splits)
        size += getsizeof(self._estimators_empty)
        size += getsizeof(self._name)
        size += getsizeof(self._n_attrs)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._n)
        size += getsizeof(self._last_n)
        size += getsizeof(self._prediction_type)
        size += getsizeof(self._children)
        size += getsizeof(self._stats)
        size += getsizeof(self._attr_types)
        size += getsizeof(self._numeric_estimator)
        size += getsizeof(self._parent)
        size += getsizeof(self._split_attr)
        size += getsizeof(self._split_value)
        size += getsizeof(self._split_gain)
        size += getsizeof(self._split_type)
        size += getsizeof(self._m_attrs)

        # this will prevent errors when computing
        # memory after calling clean_after_split
        try:
            size += getsizeof(self._dist)
            size += getsizeof(self._n_from_split)
            size += getsizeof(self._dist_from_split)
            size += getsizeof(self._attr_estimators)
            size += getsizeof(self._dropped_attrs)
            size += getsizeof(self._exhausted_attrs)
            size += sum([estimator.memory_size()
                         for estimator in self._attr_estimators])
        except TypeError:
            pass
        return size

    @property
    def parent(self):
        if self._parent is None:
            return None
        else:
            return self._parent()

    @property
    def accuracy(self):
        return self._stats['acc']

    @property
    def clean_distribution(self):
        '''
        Returns a class distribution vector ignoring
        instances coming from its parent's split
        '''

        if self._dist_from_split is None:
            return self._dist
        else:
            return self._dist - self._dist_from_split

    cdef void _update_attr_estimators(self, np.ndarray X, int y, int weight):
        '''
        Updates each attribute estimator accordingly
        '''

        cdef object estimator, x
        # flag for naive bayes
        self._estimators_empty = False
        for estimator, x in zip(self._attr_estimators, X):
            estimator.add_instance(x, y, weight)

    cdef Node move_down(self, np.ndarray X):
        if self._split_type == 'continuous':
            return self._children[self._sort_continuous(X)]
        elif self._split_type == 'nominal':
            return self._children[self._sort_nominal(X)]
        else:
            return self._children[self._sort_nominal_binary(X)]

    cdef object _sort_nominal(self, np.ndarray X):
        return X[self._split_attr]

    cdef bint _sort_nominal_binary(self, np.ndarray X):
        return X[self._split_attr] in self._le_set

    cdef bint _sort_continuous(self, np.ndarray X):
        return X[self._split_attr] <= self._split_value

    cdef void learn_from_instance(self, np.ndarray X, int y, int weight):
        # updates acc
        if self._prediction_type == 'adaptive':
            self._adaptive_update_accs(X, y, weight)
        elif self._prediction_type == 'nb':
            self._nb_update_accs(X, y, weight)
        else:
            self._mc_update_accs(X, y, weight)

        # update numeric estimators
        self._update_attr_estimators(X, y, weight)

        # add y to node count
        self._dist[y] += weight

        # increase the number of elements seen
        self._n += weight

    cdef bint all_same_class(self):
        return len(np.nonzero(self._dist)[0]) <= 1

    cdef double entropy(self):
        cdef:
            np.ndarray clean_distribution, probs
            double tot, p, entropy, v

        clean_distribution = self.clean_distribution
        tot = self._n - self._n_from_split
        entropy = 0
        for v in clean_distribution:
            if v != 0:
                p = v / tot
                entropy -= p * log2(p)
        return entropy

    cdef PossibleSplit _infogain_continuous(self, double sys_entropy,
                                            int attr):
        cdef:
            object estimator
            double total_size, split_value, le_size, gt_size, \
                entropy_left, entropy_right, average_entropy, infogain, \
                best_split_value, n, p
            np.ndarray le, gt, probs_le, probs_gt
            tuple distribution, best_distribution
            int i
            list entropies

        entropies = []
        estimator = self._attr_estimators[attr]
        total_size = estimator.total
        for split_value in estimator.possible_splits():
            le, gt = estimator.get_le_gt(split_value)
            le_size, gt_size = 0, 0
            for p in le:
                le_size += p
            for p in gt:
                gt_size += p
            distribution = (le, gt)
            entropy_left, entropy_right = 0, 0

            if le_size > 0:
                for n in le:
                    if n != 0:
                        p = n / le_size
                        entropy_left += -p * log2(p)

            if gt_size > 0:
                for n in gt:
                    if n != 0:
                        p = n / gt_size
                        entropy_right += -p * log2(p)

            # probs_le = np.fromiter([n / le_size for n in le if n != 0],
            #                         dtype=np.float64)
            # probs_gt = np.fromiter([n / gt_size for n in gt if n != 0],
            #                        dtype=np.float64)
            # entropy_left = np.sum([- p * log2(p) for p in probs_le
            #                        if p != 0])
            # entropy_right = np.sum([- p * log2(p) for p in probs_gt
            #                         if p != 0])
            average_entropy = ((entropy_left * le_size +
                                entropy_right * gt_size) / total_size)
            entropies.append((split_value, average_entropy, distribution))

        # if has_possible_split:
        if len(entropies):
            (best_split_value,
             lowest_entropy, best_distribution) = min(entropies,
                                                      key=lambda ps: ps[1])
            infogain = sys_entropy - lowest_entropy
            return PossibleSplit(attr=attr,
                                 attr_type=self._attr_types[attr],
                                 value=best_split_value,
                                 le_set=None,
                                 gain=infogain,
                                 distribution=best_distribution)

    cdef PossibleSplit _infogain_nominal(self, double sys_entropy,
                                          int attr):
        cdef:
            object estimator
            dict distribution, total_per_value, value_probs
            double total_size, weighted_entropy, weight, entropy, \
                infogain, p, tot
            np.ndarray probs

        estimator = self._attr_estimators[attr]
        distribution = estimator.get_distribution()
        total_size = estimator.total
        total_per_value = estimator.get_total_per_value()
        value_probs = estimator.get_probas()

        weighted_entropy = 0
        for value, probs in value_probs.items():
            weight = total_per_value[value]
            entropy = 0
            for p in probs:
                if p != 0:
                    entropy += -p * log2(p)
            weighted_entropy += weight * entropy
        if total_size != 0:
            weighted_entropy /= total_size

        infogain = sys_entropy - weighted_entropy
        if infogain != 0:
            return PossibleSplit(attr=attr,
                                 attr_type=self._attr_types[attr],
                                 value=None,
                                 le_set=None,
                                 gain=infogain,
                                 distribution=distribution)

    cdef PossibleSplit _infogain_nominal_binary(self, double sys_entropy,
                                                int attr):
        cdef:
            object estimator
            dict values_distribution
            list possible_nominal_values
            double total_size, le_size, gt_size, entropy_left, \
                entropy_right, average_entropy, infogain, n, p
            int i
            set le_set, best_le_set
            np.ndarray le, gt, dist, probs_le, probs_gt
            tuple distribution, best_distribution
            list entropies

        entropies = []
        estimator = self._attr_estimators[attr]
        values_distribution = estimator.get_distribution()
        total_size = estimator.total
        possible_nominal_values = list(estimator.possible_values())
        for i in range(len(possible_nominal_values) - 1):
            le_set = set(possible_nominal_values[:i])
            le, gt = np.zeros(self._n_classes), np.zeros(self._n_classes)
            for v, dist in values_distribution.items():
                if v in le_set:
                    le = le + dist
                else:
                    gt = gt + dist

            le_size, gt_size = 0, 0
            for p in le:
                le_size += p
            for p in gt:
                gt_size += p
            distribution = (le, gt)
            entropy_left, entropy_right = 0, 0
            if le_size > 0:
                for n in le:
                    if n != 0:
                        p = n / le_size
                        entropy_left += -p * log2(p)

            if gt_size > 0:
                for n in gt:
                    if n != 0:
                        p = n / gt_size
                        entropy_right += -p * log2(p)
            # probs_le = np.fromiter([n / le_size for n in le if n != 0],
            #                        dtype=np.float64)
            # probs_gt = np.fromiter([n / gt_size for n in gt if n != 0],
            #                        dtype=np.float64)
            # entropy_left = np.sum([- probs_le * np.log2(probs_le)])
            # entropy_right = np.sum([- probs_gt * np.log2(probs_gt)])
            average_entropy = ((entropy_left * le_size +
                                entropy_right * gt_size) / total_size)
            entropies.append((le_set, average_entropy, distribution))

        if len(entropies):
            (best_le_set,
             lowest_entropy, best_distribution) = min(entropies,
                                                      key=lambda ps: ps[1])
            infogain = sys_entropy - lowest_entropy
            return PossibleSplit(attr=attr,
                                 attr_type=self._attr_types[attr],
                                 value=None,
                                 le_set=best_le_set,
                                 gain=infogain,
                                 distribution=best_distribution)

    cdef double gini(self):
        cdef:
            np.ndarray clean_distribution, probs
            double tot, p, gini, v

        clean_distribution = self.clean_distribution
        tot = self._n - self._n_from_split
        if tot == 0:
            return 0
        gini = 1
        for v in clean_distribution:
            if v != 0:
                p = v / tot
                gini -= p * p
        return gini

    cdef PossibleSplit _gini_continuous(self, int attr):
        cdef:
            object estimator
            list ginis
            double total_size, split_value, le_size, gt_size, \
                gini_imp_le, gini_imp_gt, average_gini, gini, \
                best_split_value, n, p
            tuple distribution, best_distribution

        ginis = []
        estimator = self._attr_estimators[attr]
        total_size = estimator.total
        for split_value in estimator.possible_splits():
            le, gt = estimator.get_le_gt(split_value)
            le_size, gt_size = 0, 0
            for p in le:
                le_size += p
            for p in gt:
                gt_size += p
            distribution = (le, gt)

            gini_imp_le = 1
            if le_size > 0:
                for n in le:
                    p = n / le_size
                    gini_imp_le -= p * p

            gini_imp_gt = 1
            if gt_size > 0:
                for n in gt:
                    p = n / gt_size
                    gini_imp_gt -= p * p

            average_gini = ((gini_imp_le * le_size +
                            gini_imp_gt * gt_size) / total_size)
            ginis.append((split_value, average_gini, distribution))

        if len(ginis):
            (best_split_value,
             lowest_gini, best_distribution) = min(ginis,
                                                   key=lambda ps: ps[1])
            gini = 1 - lowest_gini
            return PossibleSplit(attr=attr,
                                 attr_type=self._attr_types[attr],
                                 value=best_split_value,
                                 le_set=None,
                                 gain=gini,
                                 distribution=best_distribution)

    cdef PossibleSplit _gini_nominal(self, int attr):
        cdef:
            object estimator
            dict distribution, total_per_value, value_probs
            double total_size, weighted_gini, weight, gini_imp, \
                gini, p, tot
            np.ndarray probs

        estimator = self._attr_estimators[attr]
        distribution = estimator.get_distribution()
        total_size = estimator.total
        total_per_value = estimator.get_total_per_value()
        value_probs = estimator.get_probas()

        weighted_gini = 0
        for value, probs in value_probs.items():
            weight = total_per_value[value]
            gini_imp = 1
            for p in probs:
                if p != 0:
                    gini_imp -= p * p
            weighted_gini += weight * gini_imp
        if total_size != 0:
            weighted_gini /= total_size

        gini = 1 - weighted_gini
        if gini != 0:
            return PossibleSplit(attr=attr,
                                 attr_type=self._attr_types[attr],
                                 value=None,
                                 le_set=None,
                                 gain=gini,
                                 distribution=distribution)

    cdef PossibleSplit _gini_nominal_binary(self, int attr):
        cdef:
            object estimator
            dict values_distribution
            list possible_nominal_values
            double total_size, le_size, gt_size, gini_imp_le, \
                gini_imp_gt, average_gini, gini, n, p
            int i
            set le_set, best_le_set
            np.ndarray le, gt, dist, probs_le, probs_gt
            tuple distribution, best_distribution

        ginis = []
        estimator = self._attr_estimators[attr]
        values_distribution = estimator.get_distribution()
        total_size = estimator.total
        possible_nominal_values = list(estimator.possible_values())
        for i in range(len(possible_nominal_values) - 1):
            le_set = set(possible_nominal_values[:i])
            le, gt = np.zeros(self._n_classes), np.zeros(self._n_classes)
            for v, dist in values_distribution.items():
                if v in le_set:
                    le += dist
                else:
                    gt += dist

            le_size, gt_size = 0, 0
            for p in le:
                le_size += p
            for p in gt:
                gt_size += p
            distribution = (le, gt)
            gini_imp_le, gini_imp_gt = 1, 1
            if le_size > 0:
                for n in le:
                    p = n / le_size
                    if p != 0:
                        gini_imp_le -= p * p

            if gt_size > 0:
                for n in gt:
                    p = n / gt_size
                    if p != 0:
                        gini_imp_gt -= p * p
            average_gini = ((gini_imp_le * le_size +
                            gini_imp_gt * gt_size) / total_size)
            ginis.append((le_set, average_gini, distribution))

        if len(ginis):
            (best_le_set,
             lowest_gini, best_distribution) = min(ginis,
                                                   key=lambda ps: ps[1])
            gini = 1 - lowest_gini
            return PossibleSplit(attr=attr,
                                 attr_type=self._attr_types[attr],
                                 value=None,
                                 le_set=best_le_set,
                                 gain=gini,
                                 distribution=best_distribution)

    def rank_attrs_ig(self):
        cdef:
            PossibleSplit possible_split
            int attr
            double entropy
            list rank

        entropy = self.entropy()
        rank = []
        for attr, attr_type in filter(lambda attr:
                                      attr[0] not in self._dropped_attrs,
                                      enumerate(self._attr_types)):
            if attr_type is float:
                possible_split = self._infogain_continuous(entropy, attr)
            elif self._only_binary_splits:
                possible_split = self._infogain_nominal_binary(entropy, attr)
            else:
                possible_split = self._infogain_nominal(entropy, attr)
            if possible_split is not None:
                rank.append(possible_split)
        rank.sort(key=lambda ps: ps.gain, reverse=True)
        return rank

    def rank_attrs_gini(self):
        cdef:
            PossibleSplit possible_split
            int attr
            list rank

        rank = []
        for attr, attr_type in filter(lambda attr:
                                      attr[0] not in self._dropped_attrs,
                                      enumerate(self._attr_types)):
            if attr_type is float:
                possible_split = self._gini_continuous(attr)
            elif self._only_binary_splits:
                possible_split = self._gini_nominal_binary(attr)
            else:
                possible_split = self._gini_nominal(attr)
            if possible_split is not None:
                rank.append(possible_split)
        rank.sort(key=lambda ps: ps.gain, reverse=True)
        return rank

    cdef void _drop_poor_attrs_func(self, list rank, double hb):
        cdef:
            PossibleSplit best, ps
            int i
            set removed_attrs

        best, *rest = rank
        best_gain = best.gain
        for i, ps in enumerate(rest):
            if best_gain - ps.gain > hb:
                removed_attrs = set([p.attr for p in rest[i:]])
                self._dropped_attrs |= removed_attrs
                break

    cdef int split(self, PossibleSplit possible_split):
        attr, value, le_set, gain, dist = (possible_split.attr,
                                           possible_split.value,
                                           possible_split.le_set,
                                           possible_split.gain,
                                           possible_split.distribution)
        if value is not None:
            return self._split_continuous(attr, value, gain, dist)
        elif le_set is not None:
            return self._split_nominal_binary(attr, le_set, gain, dist)
        else:
            return self._split_nominal(attr, gain, dist)

    cdef int _split_nominal(self, int attr, double gain,
                             dict distribution):
        cdef:
            Node child
            np.ndarray le_dist, gt_dist
            dict kwargs
            set extra_ex

        self._is_leaf = False
        self._split_attr = attr
        self._split_gain = gain
        self._split_type = 'nominal'

        extra_ex = set((attr, ))
        for v, dist in distribution.items():
            kwargs = dict(name=None,
                          attr_types=self._attr_types,
                          n_classes=self._n_classes,
                          numeric_estimator=self._numeric_estimator,
                          prediction_type=self._prediction_type,
                          only_binary_splits=self._only_binary_splits,
                          m_attrs=self._m_attrs,
                          exhausted_attrs=self._exhausted_attrs | extra_ex,
                          parent=self,
                          distribution=dist)
            child = type(self)(**kwargs)
            self._children[v] = child
        return len(distribution.keys())

    cdef int _split_nominal_binary(self, int attr, set le_set, double gain,
                                   tuple distribution):
        cdef:
            Node left_child, right_child
            np.ndarray le_dist, gt_dist
            dict kwargs_left, kwargs_right
            set extra_ex

        self._is_leaf = False
        self._split_attr = attr
        self._split_gain = gain
        self._split_type = 'nominal_binary'
        self._le_set = le_set

        extra_ex = set((attr, ))
        le_dist, gt_dist = distribution

        # less-equal
        kwargs_left = dict(name=None,
                           attr_types=self._attr_types,
                           n_classes=self._n_classes,
                           numeric_estimator=self._numeric_estimator,
                           prediction_type=self._prediction_type,
                           only_binary_splits=self._only_binary_splits,
                           m_attrs=self._m_attrs,
                           exhausted_attrs=(self._exhausted_attrs |
                                            extra_ex),
                           parent=self,
                           distribution=le_dist)
        left_child = type(self)(**kwargs_left)
        self._children[True] = left_child

        # greater than
        kwargs_right = dict(name=None,
                            attr_types=self._attr_types,
                            n_classes=self._n_classes,
                            numeric_estimator=self._numeric_estimator,
                            prediction_type=self._prediction_type,
                            only_binary_splits=self._only_binary_splits,
                            m_attrs=self._m_attrs,
                            exhausted_attrs=(self._exhausted_attrs |
                                             extra_ex),
                            parent=self,
                            distribution=gt_dist)
        right_child = type(self)(**kwargs_right)
        self._children[False] = right_child

        return 2

    cdef int _split_continuous(self, int attr, double value,
                               double gain, tuple distribution):
        cdef:
            Node left_child, right_child
            np.ndarray le_dist, gt_dist
            dict kwargs_left, kwargs_right
            set extra_ex

        self._is_leaf = False
        self._split_attr = attr
        self._split_value = value
        self._split_gain = gain
        self._split_type = 'continuous'

        le_dist, gt_dist = distribution
        # less-equal
        kwargs_left = dict(name=None,
                           attr_types=self._attr_types,
                           n_classes=self._n_classes,
                           numeric_estimator=self._numeric_estimator,
                           prediction_type=self._prediction_type,
                           only_binary_splits=self._only_binary_splits,
                           m_attrs=self._m_attrs,
                           exhausted_attrs=self._exhausted_attrs,
                           parent=self,
                           distribution=le_dist)
        left_child = type(self)(**kwargs_left)
        self._children[True] = left_child

        # greater than
        kwargs_right = dict(name=None,
                            attr_types=self._attr_types,
                            n_classes=self._n_classes,
                            numeric_estimator=self._numeric_estimator,
                            prediction_type=self._prediction_type,
                            only_binary_splits=self._only_binary_splits,
                            m_attrs=self._m_attrs,
                            exhausted_attrs=self._exhausted_attrs,
                            parent=self,
                            distribution=gt_dist)
        right_child = type(self)(**kwargs_right)
        self._children[False] = right_child

        return 2

    cdef void clean_memory(self):
        PyObject_DelAttr(self, '_dropped_attrs')
        PyObject_DelAttr(self, '_exhausted_attrs')
        PyObject_DelAttr(self, '_attr_estimators')
        PyObject_DelAttr(self, '_dist_from_split')
        PyObject_DelAttr(self, '_n_from_split')
        PyObject_DelAttr(self, '_dist')

    cdef np.ndarray mc_predict_proba(self):
        cdef:
            np.ndarray probs
            double tot, p

        if self._n == 0:
            probs = np.empty(self._n_classes)
            probs[:] = 1.0 / self._n_classes
            return probs
        tot = 0
        for p in self._dist:
            tot += p
        if tot != 0:
            return self._dist / tot
        else:
            return self._dist

    cdef int mc_predict(self):
        if self._n == 0:
            return np.random.randint(0, self._n_classes)

        cdef int yhat = -1, i
        cdef double v = -INFINITY, p
        cdef np.ndarray probs

        probs = self._dist
        for i in range(probs.shape[0]):
            p = probs[i]
            if p > v:
                v = p
                yhat = i
        return yhat
        # return np.argmax(self._dist)

    cdef np.ndarray nb_predict_proba(self, np.ndarray X):
        cdef:
            object estimator, x
            np.ndarray probs, exp_probs
            int y, i
            double tot

        probs = self.mc_predict_proba()
        if not self._estimators_empty:
            for y in range(self._n_classes):
                for x, estimator in zip(X, self._attr_estimators):
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

    cdef int nb_predict(self, np.ndarray X):
        cdef int yhat = -1, i
        cdef double v = -INFINITY, p
        cdef np.ndarray probs

        probs = self.nb_predict_proba(X)
        for i in range(probs.shape[0]):
            p = probs[i]
            if p > v:
                v = p
                yhat = i

        # yhat = np.argmax(self.nb_predict_proba(X))
        return yhat

    cdef np.ndarray nb_predict_log_proba(self, np.ndarray X):
        cdef:
            object x, estimator
            np.ndarray probs
            int y, i
            double tot

        probs = np.log(self.mc_predict_proba())
        if not self._estimators_empty:
            for y in range(self._n_classes):
                for x, estimator in zip(X, self._attr_estimators):
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

    cdef int nb_predict_log(self, np.ndarray X):
        cdef int yhat = -1, i
        cdef double v = -INFINITY, p
        cdef np.ndarray probs

        probs = self.nb_predict_proba(X)
        for i in range(probs.shape[0]):
            p = probs[i]
            if p > v:
                v = p
                yhat = i
        return yhat
        # return np.argmax(self.nb_predict_log_proba(X))

    cdef np.ndarray predict_probs(self, np.ndarray X):
        cdef np.ndarray probs

        if self._prediction_type == 'adaptive':
            probs = self.adaptive_predict_proba(X)
        elif self._prediction_type == 'nb':
            probs = self.nb_predict_proba(X)
        else:
            probs = self.mc_predict_proba()
        return probs

    cdef void _update_accs(self, int success, int weight):
        cdef:
            double last_acc
            int n

        self._stats['elements_seen'] += weight
        n = self._stats['elements_seen']
        last_acc = self._stats['acc']
        self._stats['acc'] += (success - last_acc) / n
        self._stats['elements_seen'] += weight

    cdef tuple _mc_update_accs(self, np.ndarray X, int y, int weight):
        cdef:
            np.ndarray probs
            int yhat = -1, i, success
            double v = -INFINITY, p

        probs = self.mc_predict_proba()
        for i in range(probs.shape[0]):
            p = probs[i]
            if p > v:
                v = p
                yhat = i

        # yhat = np.argmax(probs)
        success = 1 if yhat == y else 0
        self._update_accs(success, weight)
        return probs, yhat

    cdef tuple _nb_update_accs(self, np.ndarray X, int y, int weight):
        cdef:
            np.ndarray probs
            int yhat = -1, i, success
            double v = -INFINITY, p

        probs = self.nb_predict_proba(X)
        for i in range(probs.shape[0]):
            p = probs[i]
            if p > v:
                v = p
                yhat = i

        # yhat = np.argmax(probs)
        success = 1 if yhat == y else 0
        self._update_accs(success, weight)
        return probs, yhat

    cdef bint _nb_is_better(self):
        return self._stats['missed_mc'] >= self._stats['missed_nb']

    cdef tuple _adaptive_update_accs(self, np.ndarray X, int y, int weight):
        cdef:
            np.ndarray probs_mc, probs_nb, probs
            int y_mc = -1, y_nb = -1, yhat = -1, success, i
            double v = -INFINITY, p

        probs_mc = self.mc_predict_proba()
        probs_nb = self.nb_predict_proba(X)

        for i in range(probs_mc.shape[0]):
            p = probs_mc[i]
            if p > v:
                v = p
                y_mc = i

        # y_mc = np.argmax(probs_mc)
        v = - INFINITY
        for i in range(probs_nb.shape[0]):
            p = probs_nb[i]
            if p > v:
                v = p
                y_nb = i

        # y_nb = np.argmax(probs_nb)
        if self._nb_is_better():
            yhat = y_nb
            probs = probs_nb
        else:
            yhat = y_mc
            probs = probs_mc
        success = 1 if yhat == y else 0
        self._update_accs(success, weight)
        if y_mc != y:
            self._stats['missed_mc'] += weight
        if y_nb != y:
            self._stats['missed_nb'] += weight
        return probs, yhat

    cdef int adaptive_predict(self, np.ndarray X):
        cdef int yhat

        if self._nb_is_better():
            yhat = self.nb_predict(X)
        else:
            yhat = self.mc_predict()
        return yhat

    cdef np.ndarray adaptive_predict_proba(self, np.ndarray X):
        cdef np.ndarray probs

        if self._nb_is_better():
            probs = self.nb_predict_proba(X)
        else:
            probs = self.mc_predict_proba()
        return probs

    cdef tuple adaptive_predict_proba_with_type(self, np.ndarray X):
        cdef:
            np.ndarray probs
            str type_

        if self._nb_is_better():
            probs, type_ = self.nb_predict_proba(X), 'nb'
        else:
            probs, type_ = self.mc_predict_proba(), 'mc'
        return probs, type_


cdef class VFDT:
    """
    This is an implementation of the VFDT described by Domingos (add_date)
    Additional info:
        - this implementation does not start the number of elements
          seem by a node with the number of elements
          coming from its parent's split (unlike MOA),
          since it affects the hoeffding bound computation;
    """

    version = '1.0'

    def __init__(self, attr_types: Iterable, int n_classes, *,
                 int gp=200,
                 double delta=1e-5, double tiebreaker=0.05,
                 object numeric_estimator=ClassGaussianEstimator,
                 str prediction_type='adaptive',
                 str split_criterion='infogain',
                 bint only_binary_splits=False,
                 bint drop_poor_attrs=True,
                 bint clean_after_split=True,
                 m_attrs: Optional[int]=None,
                 object node_class=Node):

        # assert issubclass(numeric_estimator, NumericEstimatorABC)
        assert prediction_type in {'mc', 'nb', 'adaptive'}
        assert split_criterion in {'infogain', 'gini'}

        self._attr_types = attr_types
        self._n_attrs = len(attr_types)
        self._n_classes = n_classes
        self._grace_period = gp
        self._delta = delta
        self._tiebreaker = tiebreaker
        self._split_confidence = 1 - delta
        self._numeric_estimator = numeric_estimator
        self._prediction_type = prediction_type
        self._split_criterion = split_criterion
        if split_criterion == 'infogain':
            self._rank_function = 'rank_attrs_ig'
            self._split_crit_range = log2(self._n_classes)
        else:
            self._rank_function = 'rank_attrs_gini'
            self._split_crit_range = 1.0
        self._only_binary_splits = only_binary_splits
        self._drop_poor_attrs = drop_poor_attrs
        self._clean_after_split = clean_after_split
        self._m_attrs = m_attrs

        # Initiate the root
        self._root = node_class(name=0, attr_types=attr_types,
                                n_classes=n_classes,
                                numeric_estimator=numeric_estimator,
                                prediction_type=self._prediction_type,
                                only_binary_splits=self._only_binary_splits,
                                m_attrs=m_attrs, exhausted_attrs=None)
        self._next_name = 1
        self._leaves = {0: self._root}
        self._stats = TreeStats(self._n_classes)

    cdef double _memory_size(self):
        """
        Returns the amount of RAM memory used in B
        """
        cdef:
            double size, leaf_size
            int n_leaves, split_nodes
            Node node

        size = getsizeof(self)
        size += getsizeof(self._attr_types)
        size += getsizeof(self._n_attrs)
        size += getsizeof(self._n_classes)
        size += getsizeof(self._grace_period)
        size += getsizeof(self._delta)
        size += getsizeof(self._tiebreaker)
        size += getsizeof(self._split_confidence)
        size += getsizeof(self._prediction_type)
        size += getsizeof(self._split_crit_range)
        size += getsizeof(self._split_criterion)
        size += getsizeof(self._numeric_estimator)
        size += getsizeof(self._rank_function)
        size += getsizeof(self._only_binary_splits)
        size += getsizeof(self._drop_poor_attrs)
        size += getsizeof(self._clean_after_split)
        size += getsizeof(self._m_attrs)
        size += getsizeof(self._root)
        size += getsizeof(self._next_name)
        size += getsizeof(self._leaves)
        size += getsizeof(self._stats)

        size += self._stats.memory_size()

        n_leaves = self._stats['n_leaves']
        split_nodes = self._stats['n_nodes'] - n_leaves

        if split_nodes == 0:
            node = self._root
            size += node._memory_size()
        else:
            node = self._root
            size += node._memory_size() * split_nodes
            node = next(iter(self._leaves.values()))
            leaf_size = node._memory_size()
            size += leaf_size * n_leaves
        return size

    cdef void _update_leaves_hash(self, tuple names, tuple new_leaves):
        cdef:
            Node leaf
            int name

        for name, leaf in zip(names, new_leaves):
            leaf._name = name
            self._leaves[name] = leaf

    cdef void _update_split_stats(self, int new_nodes):
        self._stats['splits'] += 1
        self._stats['n_nodes'] += new_nodes
        self._stats['n_leaves'] += new_nodes - 1

    cdef Node _sort_to_leaf(self, np.ndarray X):
        cdef Node node

        node = self._root
        while not node._is_leaf:
            node = node.move_down(X)
        return node

    cdef double _hoeffding_bound(self, double delta, double R, double n):
        cdef double R2

        R2 = R * R
        return sqrt((R2 * log(1 / delta)) / (2 * n))

    cdef bint _can_split(self, list rank, double hb, Node leaf):
        cdef bint vfdt_cond

        # very specific case where all the instances are from different classes
        # but all attributes have the same value
        # should never happen #
        # if any([isfinite(ps.gain) for ps in rank]):
        #     vfdt_cond = self._can_split_vfdt(rank, hb)
        #     return vfdt_cond
        # return False
        return self._can_split_vfdt(rank, hb)

    cdef bint _can_split_vfdt(self, list rank, double hb):
        cdef:
            PossibleSplit best, sec_best
            double best_gain, secbest_gain

        if len(rank) == 1:
            return True
        best, sec_best = rank[:2]
        best_gain = best.gain
        secbest_gain = sec_best.gain
        return best_gain - secbest_gain > hb or hb < self._tiebreaker

    cdef void _split_leaf(self, PossibleSplit split, Node leaf):
        cdef:
            int n_child
            tuple names

        # no longer a leaf
        del self._leaves[leaf._name]

        # split leaf and get number of children
        n_child = leaf.split(split)
        names = tuple(range(self._next_name, self._next_name + n_child))

        self._next_name += n_child

        # this is used to save the new leaves
        self._update_leaves_hash(names, tuple(leaf._children.values()))

        # update split statistics
        self._update_split_stats(n_child)

    cdef void _reset_leaf(self, Node leaf):
        leaf._reset()

    cdef void _train(self, np.ndarray X, int y, int weight=1):
        cdef:
            Node leaf
            double gp, current_n, last_n, hb
            list rank
            PossibleSplit split

        if weight == 0:
            return

        leaf = self._sort_to_leaf(X)

        leaf.learn_from_instance(X, y, weight)
        # all attributes were exhausted for that leaf

        # the chances of this happening are very low and all attributes
        # need to be nominal, but this may save some computational time
        # (and is better than setting leaf._last_n to INFINITY)
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
                                               R=self._split_crit_range,
                                               n=current_n)
                    if self._can_split(rank, hb, leaf):
                        split = rank[0]
                        self._split_leaf(split, leaf)
                        # Just delete the estimators to free up some memory
                        if self._clean_after_split:
                            leaf.clean_memory()
                    elif self._drop_poor_attrs:
                        leaf._drop_poor_attrs_func(rank, hb)

    cdef int _predict(self, np.ndarray X):
        cdef:
            Node leaf
            int yhat

        leaf = self._sort_to_leaf(X)
        if self._prediction_type == 'adaptive':
            yhat = leaf.adaptive_predict(X)
        elif self._prediction_type == 'nb':
            yhat = leaf.nb_predict(X)
        else:
            yhat = leaf.mc_predict()
        return yhat

    cdef np.ndarray _predict_proba(self, np.ndarray X):
        cdef:
            Node leaf
            np.ndarray probs

        leaf = self._sort_to_leaf(X)
        if self._prediction_type == 'mc':
            probs = leaf.mc_predict_proba()
        elif self._prediction_type == 'nb':
            probs = leaf.nb_predict_proba(X)
        elif self._prediction_type == 'adaptive':
            probs = leaf.adaptive_predict_proba(X)
        return probs

    cdef tuple _predict_proba_with_type(self, np.ndarray X):
        cdef:
            Node leaf
            np.ndarray probs
            str type_

        leaf = self._sort_to_leaf(X)
        if self._prediction_type == 'mc':
            probs, type_ = leaf.mc_predict_proba(), 'mc'
        elif self._prediction_type == 'nb':
            probs, type_ = leaf.nb_predict_proba(X), 'nb'
        elif self._prediction_type == 'adaptive':
            probs, type_ = leaf.adaptive_predict_proba_with_type(X)
        return probs, type_

    cdef tuple _predict_both(self, np.ndarray X):
        cdef:
            Node leaf
            int yhat_mc, yhat_nb, yhat

        leaf = self._sort_to_leaf(X)
        yhat_mc = leaf.mc_predict()
        yhat_nb = leaf.nb_predict(X)
        if leaf.nb_is_better():
            yhat = yhat_nb
        else:
            yhat = yhat_mc
        return yhat, yhat_mc, yhat_nb

    cdef TreeStats _get_stats(self):
        return self._stats

    def memory_size(self):
        return self._memory_size()

    def train(self, np.ndarray X, int y, int weight=1):
        return self._train(X, y, weight)

    def predict(self, np.ndarray X):
        return self._predict(X)

    def predict_proba(self, np.ndarray X):
        return self._predict_proba(X)

    def predict_proba_with_type(self, np.ndarray X):
        return self._predict_proba_with_type(X)

    def predict_both(self, np.ndarray X):
        return self._predict_both(X)

    def get_stats(self):
        return self._get_stats()
