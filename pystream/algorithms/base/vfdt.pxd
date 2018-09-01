cimport numpy as np
from .statistics.tree_stats cimport TreeStats


cdef class PossibleSplitC:
    cdef:
        readonly int attr
        readonly object attr_type, value
        readonly set le_set
        readonly double gain
        readonly object distribution


cdef class Node:
    cdef:
        object __weakref__
        public bint active, _is_leaf
        bint _only_binary_splits, _estimators_empty
        public object _name
        int _n_attrs, _n_classes
        public double _n,  _last_n
        str _prediction_type
        public list _attr_estimators
        public np.ndarray _dist
        public dict _children, _stats
        list _attr_types
        object _numeric_estimator
        object _parent
        public object _exhausted_attrs
        public object _dropped_attrs
        public object _start_dist
        public object _start_n
        object _split_attr
        object _split_value
        object _split_gain
        object _split_type
        object _m_attrs
        object _le_set

    cdef void _reset(self)

    cdef double _memory_size(self)

    cdef void _update_attr_estimators(self, np.ndarray X, int y, int weight)

    cdef Node move_down(self, np.ndarray X)

    cdef object _sort_nominal(self, np.ndarray X)

    cdef bint _sort_nominal_binary(self, np.ndarray X)

    cdef bint _sort_continuous(self, np.ndarray X)

    cdef void learn_from_instance(self, np.ndarray X, int y, int weight)

    cdef bint all_same_class(self)

    cdef double entropy(self)

    cdef PossibleSplitC _infogain_continuous(self, double sys_entropy,
                                             int attr)

    cdef PossibleSplitC _infogain_nominal(self, double sys_entropy, int attr)

    cdef PossibleSplitC _infogain_nominal_binary(self, double sys_entropy,
                                                 int attr)

    cdef double gini(self)

    cdef PossibleSplitC _gini_gain_continuous(self, int attr)

    cdef PossibleSplitC _gini_gain_nominal(self, int attr)

    cdef PossibleSplitC _gini_gain_nominal_binary(self, int attr)

    cdef void _drop_poor_attrs_func(self, list rank, double hb)

    cdef int split(self, PossibleSplitC possible_split)

    cdef int _split_nominal(self, int attr, double gain,
                             dict distribution)

    cdef int _split_nominal_binary(self, int attr, set le_set, double gain,
                                   tuple distribution)

    cdef int _split_continuous(self, int attr, double value,
                               double gain, tuple distribution)

    cdef void clean_memory(self)

    cdef np.ndarray mc_predict_proba(self)

    cdef int mc_predict(self)

    cdef np.ndarray nb_predict_proba(self, np.ndarray X)

    cdef int nb_predict(self, np.ndarray X)

    cdef np.ndarray nb_predict_log_proba(self, np.ndarray X)

    cdef int nb_predict_log(self, np.ndarray X)

    cdef np.ndarray predict_probs(self, np.ndarray X)

    cdef void _update_accs(self, int success, int weight)

    cdef tuple _mc_update_accs(self, np.ndarray X,
                               int y, int weight)

    cdef tuple _nb_update_accs(self, np.ndarray X, int y, int weight)

    cdef bint _nb_is_better(self)

    cdef tuple _adaptive_update_accs(self, np.ndarray X, int y, int weight)

    cdef int adaptive_predict(self, np.ndarray X)

    cdef np.ndarray adaptive_predict_proba(self, np.ndarray X)

    cdef tuple adaptive_predict_proba_with_type(self, np.ndarray X)


cdef class VFDT:
    cdef:
        object _attr_types
        int _n_attrs, _n_classes, _grace_period
        double _split_confidence, _tiebreaker, _delta, _split_crit_range
        str _prediction_type, _split_criterion
        object _numeric_estimator
        object _rank_function
        bint _only_binary_splits, _drop_poor_attrs, _clean_after_split
        object _m_attrs
        Node _root
        int _next_name
        public dict _leaves
        public object _stats

    cdef double _memory_size(self)

    cdef void _update_leaves_hash(self, tuple names, tuple new_leaves)

    cdef void _update_split_stats(self, int new_nodes)

    cdef Node _sort_to_leaf(self, np.ndarray X)

    cdef double _hoeffding_bound(self, double delta, double R, double n)

    cdef bint _can_split(self, list rank, double hb, Node leaf)

    cdef bint _can_split_vfdt(self, list rank, double hb)

    cdef void _split_leaf(self, PossibleSplitC split, Node leaf)

    cdef void _reset_leaf(self, Node leaf)

    cdef void _train(self, np.ndarray X, int y, int weight=*)

    cdef int _predict(self, np.ndarray X)

    cdef np.ndarray _predict_proba(self, np.ndarray X)

    cdef tuple _predict_proba_with_type(self, np.ndarray X)

    cdef tuple _predict_both(self, np.ndarray X)

    cdef TreeStats _get_stats(self)
