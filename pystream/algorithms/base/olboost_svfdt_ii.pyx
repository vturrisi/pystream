# cython: boundscheck=False, wraparound=False

from vfdt cimport Node
from olboost_svfdt cimport OLBoostSVFDT


cdef class OLBoostSVFDT_II(OLBoostSVFDT):
    cpdef bint _can_split_svfdt(self, Node leaf, double best_gain):
        cdef bint split, leaves_imp_constraint, imp_constraint, \
            gain_constraint, nseen_constraint
        cdef double min_svfdt2_gain, min_svfdt2_imp, min_imp_leaves, \
            min_imp, min_gain, min_n, impurity, nseen

        if self._split_criterion == 'infogain':
            impurity = leaf.entropy()
        else:
            impurity = leaf.gini()
        nseen = leaf._n
        split = False
        # SVFDT_II split test
        min_svfdt2_gain = self._gain_stats.mean_plus_std()
        min_svfdt2_imp = self._imp_stats.mean_plus_std()
        if best_gain >= min_svfdt2_gain or impurity >= min_svfdt2_imp:
            split = True
        else:
            if self._split_criterion == 'infogain':
                min_imp_leaves = self._min_entropy_leaves(self._imp_factor)
            else:
                min_imp_leaves = self._min_gini_leaves(self._imp_factor)

            min_imp = self._imp_stats.mean_plus_factor_std(self._imp_factor)
            min_gain = self._gain_stats.mean_plus_factor_std(self._gain_factor)
            min_n = self._n_stats.mean_plus_factor_std(self._n_factor)

            # contraints
            leaves_imp_constraint = impurity >= min_imp_leaves
            imp_constraint = impurity >= min_imp
            gain_constraint = best_gain >= min_gain
            nseen_constraint = nseen >= min_n

            split = all((leaves_imp_constraint, imp_constraint,
                         gain_constraint, nseen_constraint))

        # update all statistics
        self._imp_stats.update(impurity)
        self._gain_stats.update(best_gain)
        self._n_stats.update(nseen)

        # this is just to compute the number of splits blocked by the SVFDT
        if not split:
            self._stats['splits_blocked'] += 1
        return split
