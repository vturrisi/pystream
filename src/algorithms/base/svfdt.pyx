# cython: boundscheck=False, wraparound=False

cimport numpy as np
from .vfdt cimport VFDT, Node

import numpy as np
from sys import getsizeof
from .numeric_estimators.gaussian_estimator import ClassGaussianEstimator
from .statistics.value_stats import NormalDistValueStats
from .vfdt import VFDT, Optional, Node, Iterable



cdef class SVFDT(VFDT):
    def __init__(self, attr_types: Iterable, int n_classes, *,
                 int gp=200,
                 double delta=1e-5, double tiebreaker=0.05,
                 double imp_factor=-1,
                 double gain_factor=-1,
                 double n_factor=0,
                 object numeric_estimator=ClassGaussianEstimator,
                 str prediction_type='adaptive',
                 str split_criterion='infogain',
                 bint only_binary_splits=False,
                 bint drop_poor_attrs=True,
                 bint clean_after_split=True,
                 m_attrs: Optional[int]=None,
                 object node_class=Node):

        VFDT.__init__(self, attr_types=attr_types,
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

        self._imp_factor = imp_factor
        self._gain_factor = gain_factor
        self._n_factor = n_factor

        self._imp_stats = NormalDistValueStats()
        self._gain_stats = NormalDistValueStats()
        self._n_stats = NormalDistValueStats()

    cdef double _memory_size(self):
        """
        Returns the amount of RAM memory used in kB
        """
        cdef double size

        size = VFDT._memory_size(self)
        size += self._imp_stats.memory_size()
        size += self._gain_stats.memory_size()
        size += self._n_stats.memory_size()
        size += getsizeof(self._imp_factor)
        size += getsizeof(self._gain_factor)
        size += getsizeof(self._n_factor)

        return size

    cdef double _min_entropy_leaves(self, double factor):
        cdef np.ndarray entropies
        cdef Node leaf

        entropies = np.fromiter([leaf.entropy()
                                 for leaf
                                 in self._leaves.values()],
                                dtype=np.float64)
        return np.mean(entropies) + factor * np.std(entropies)

    cdef double _min_gini_leaves(self, double factor):
        cdef np.ndarray ginis
        cdef Node leaf

        ginis = np.fromiter([leaf.gini() for leaf
                             in self._leaves.values()],
                            dtype=np.float64)
        return np.mean(ginis) + factor * np.std(ginis)

    cdef bint _can_split(self, list rank, double hb, Node leaf):
        cdef double best_gain
        cdef bint vfdt_cond, svfdt_cond
        vfdt_cond = self._can_split_vfdt(rank, hb)
        best_gain = rank[0].gain
        if vfdt_cond:
            svfdt_cond = self._can_split_svfdt(leaf, best_gain)
        return vfdt_cond and svfdt_cond

    cdef bint _can_split_svfdt(self, Node leaf, double best_gain):
        cdef double impurity, nseen
        cdef double min_imp_leaves, min_imp, min_gain, min_n
        cdef bint leaves_imp_constraint, imp_constraint, gain_constraint, \
            nseen_constraint, svfdt_constraints
        cdef object imp_func


        nseen = leaf._n
        # thresholds
        if self._split_criterion == 'infogain':
            impurity = leaf.entropy()
            min_imp_leaves = self._min_entropy_leaves(self._imp_factor)
        else:
            impurity = leaf.gini()
            min_imp_leaves = self._min_gini_leaves(self._imp_factor)

        min_imp = self._imp_stats.mean_plus_factor_std(self._imp_factor)
        min_gain = self._gain_stats.mean_plus_factor_std(self._gain_factor)
        min_n = self._n_stats.mean_plus_factor_std(self._n_factor)

        # contraints
        leaves_imp_constraint = impurity >= min_imp_leaves
        imp_constraint = impurity >= min_imp
        gain_constraint = best_gain >= min_gain
        nseen_constraint = nseen >= min_n

        # nseen_constraint = True
        svfdt_constraints = all((leaves_imp_constraint, imp_constraint,
                                 gain_constraint, nseen_constraint))

        # update all statistics
        self._imp_stats.update(impurity)
        self._gain_stats.update(best_gain)
        self._n_stats.update(nseen)

        # this is just to compute the number of splits blocked by the SVFDT
        if svfdt_constraints is False:
            self._stats['splits_blocked'] += 1
        return svfdt_constraints
