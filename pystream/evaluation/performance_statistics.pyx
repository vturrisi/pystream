#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np


cdef class ConfusionMatrix:
    cdef:
        public np.ndarray confusion_matrix

    def __init__(self, n_classes):
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def __getitem__(self, key):
        if key not in self.confusion_matrix:
            return None
        return self.confusion_matrix[key]

    def __setitem__(self, key, value):
        self.confusion_matrix[key] = value


cdef class PerformanceStats:
    cdef:
        int _n_classes
        public dict _stats

    def __init__(self, n_classes):
        self._n_classes = n_classes
        self._stats = {'n': 0, 'right_preds': 0, 'wrong_preds': 0,
                       'acc_history': [],
                       'acc_per_class_history': {i: [] for i
                                                 in range(n_classes)},
                       'cm': np.zeros((n_classes, n_classes)),
                       'train_time': None,
                       'instances_seen_per_class': np.zeros(n_classes),
                       }

    @property
    def confusion_matrix(self):
        return self._stats['cm']

    @property
    def right_preds(self):
        return self._stats['cm'].diagonal().sum()

    @property
    def wrong_preds(self):
        return self._stats['cm'].sum() - self.right_preds

    @property
    def accuracy(self):
        cdef double total, right
        total = self._stats['n']
        right = self.confusion_matrix.diagonal().sum()
        return right / total if total else -1

    cdef double recall_class(self, class_):
        cdef double total, right
        total = self.confusion_matrix[class_].sum()
        right = self.confusion_matrix[class_, class_].sum()
        return right / total if total else -1

    cdef double precision_class(self, class_):
        cdef double total, right
        total = self.confusion_matrix[:, class_].sum()
        right = self.confusion_matrix[class_, class_].sum()
        return right / total if total else -1

    def __getitem__(self, key):
        if key not in self._stats:
            return None
        return self._stats[key]

    def __setitem__(self, key, value):
        self._stats[key] = value
