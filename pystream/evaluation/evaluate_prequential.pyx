# cython: boundscheck=False

import numpy as np
cimport numpy as np
cimport cython
import time
from .performance_statistics import PerformanceStats
from collections import deque
import io
import shutil


cdef class EvaluatePrequential:
    cdef int _n_classes
    cdef _algorithm
    cdef str _algorithm_type, _name
    cdef _alg_specific_stats
    cdef public stats
    cdef np.ndarray last_scores
    cdef int last_pos
    cdef bint FIRST_CYCLE
    cdef object _memory_log, _log

    MAXDIGITS = 4
    MAXDIGITS_TIME = 1

    def __init__(self, n_classes, algorithm, algorithm_type):
        self._n_classes = n_classes
        self._algorithm = algorithm
        self._algorithm_type = algorithm_type
        if algorithm_type == 'tree':
            self._alg_specific_stats = self._tree_stats
        elif algorithm_type == 'ensemble':
            self._alg_specific_stats = self._ensemble_stats
        else:
            self._alg_specific_stats = self._other_stats
        self._name = algorithm.__class__.__name__
        self.stats = PerformanceStats(self._n_classes)

        self._memory_log = io.StringIO()
        self._log = io.StringIO()

    cpdef tuple _base_stats(self, double elapsed_time, double current_acc):
        cdef double acc_mean
        stats = self.stats
        acc_mean = stats['acc_history'][-1]
        # rounding is kinda useless
        # current_acc = round(current_acc, self.MAXDIGITS)
        # acc_mean = round(stats['acc_history'][-1], self.MAXDIGITS)
        # elapsed_time = round(elapsed_time, 0)
        return current_acc, acc_mean, elapsed_time

    cpdef tuple _tree_stats(self):
        cdef int n_nodes, splits

        n_nodes = self._algorithm.get_stats()['n_nodes']
        splits = self._algorithm.get_stats()['splits']
        return n_nodes, splits

    cpdef _ensemble_stats(self):
        cdef str n_nodes, splits

        stats = self._algorithm.get_stats()
        n_nodes = '-'.join([str(stat['n_nodes']) for stat in stats])
        splits = '-'.join([str(stat['splits']) for stat in stats])
        return n_nodes, splits

    cpdef tuple _other_stats(self):
        return None, None

    cpdef _print_debug(self, int i, double elapsed_time, double current_acc):
        cdef:
            double acc_mean, size
            n_nodes, splits

        acc_mean = self.stats['acc_history'][-1]
        n_nodes, splits = self._alg_specific_stats()
        size = self._algorithm.memory_size()
        print('({}) i: {} | current_acc: {:.4f} | mean_acc: {:.4f}'
              ' | time: {:.4f}s | nodes: {} | splits: {}'
              ' | memory: {} bytes'.format(self._name, i, current_acc, acc_mean,
                                           elapsed_time, n_nodes, splits,
                                           size / 1024))

    cpdef _update_log(self, int i, double elapsed_time, double current_acc):
        cdef:
            double acc_mean, size
            n_nodes, splits

        acc_mean = self.stats['acc_history'][-1]
        n_nodes, splits = self._alg_specific_stats()
        # this is really costly
        size = self._algorithm.memory_size()
        line = ('{},{},{},{},{},{},{},{}\n'.format(self._name, i, current_acc,
                                                   acc_mean, elapsed_time,
                                                   n_nodes, splits, size))
        self._log.write(line)

    cpdef _write_log(self, str log_file, int window_size):
        with open(log_file, 'w') as fd:
            fd.write('name,i,acc_last_{},acc_mean,time(s),n_nodes,'
                     'splits,memory_size\n'.format(window_size))
            self._log.seek(0)
            shutil.copyfileobj(self._log, fd)

    cpdef _write_cm_log(self, str log_file):
        cdef str fname = log_file.split('.csv')[0]
        with open('{}_cm.txt'.format(fname), 'w') as f:
            f.write('linha: real | coluna: predito\n')
            f.write(np.array_str(self.stats['cm']))

    cpdef _compute_le_acc(self, y, yhat, window_size):
        cdef:
            double current_acc

        self.last_scores[self.last_pos] = 1 if yhat == y else 0
        self.last_pos = (self.last_pos + 1) % window_size

        # accuracy on last elements
        if self.FIRST_CYCLE and self.last_pos == 0:
            self.FIRST_CYCLE = False
        if self.FIRST_CYCLE:
            current_acc = np.mean(self.last_scores[:self.last_pos])
        else:
            current_acc = np.mean(self.last_scores)
        return current_acc

    cdef _update_pred_stats(self, int y, int yhat):
        cdef:
            int prediction
            double n, last_acc, acc

        stats = self.stats
        stats['cm'][y, yhat] += 1
        if yhat == y:
            prediction = 1
        else:
            prediction = 0
        # update general accuracy
        n = stats['n']
        if n > 0:
            last_acc = stats['acc_history'][-1]
            acc = (last_acc * n + prediction) / (n + 1)
            stats['acc_history'].append(acc)
        else:
            stats['acc_history'].append(prediction)
        # update accuracy given the class y
        n = stats['instances_seen_per_class'][y]
        if n > 0:
            last_acc = stats['acc_per_class_history'][y][-1]
            acc = (last_acc * n + prediction) / (n + 1)
            stats['acc_per_class_history'][y].append(acc)
        else:
            stats['acc_per_class_history'][y].append(prediction)
        stats['n'] += 1
        stats['instances_seen_per_class'][y] += 1

    def train_test_prequential_cm(self, stream, bint debug=False,
                                  int window_size=100, int frequency=1000,
                                  log_file=None):
        cdef:
            np.ndarray X
            int y, row_i, yhat, weight
            double start_time, elapsed_time, current_acc

        self.last_scores = np.zeros(window_size, dtype=np.float64)
        self.last_pos = 0
        self.FIRST_CYCLE = True

        # learn first example
        row_i, (X, y) = next(stream)

        # start computing time after loading dataset
        start_time = time.time()
        weight = 1
        self._algorithm.train(X, y, weight)
        # learn from other examples
        for row_i, (X, y) in stream:
            yhat = self._algorithm.predict(X)
            self._update_pred_stats(y, yhat)
            current_acc = self._compute_le_acc(y, yhat, window_size)
            elapsed_time = time.time() - start_time

            if row_i % frequency == 0:
                if debug:
                    self._print_debug(row_i, elapsed_time, current_acc)
                if log_file:
                    self._update_log(row_i, elapsed_time, current_acc)

            self._algorithm.train(X, y, weight)
        elapsed_time = time.time() - start_time
        stats = self.stats
        stats['train_time'] = elapsed_time
        if log_file:
            self._update_log(row_i, elapsed_time, current_acc)
            self._write_log(log_file, window_size)
            self._write_cm_log(log_file)

    def train_test_prequential_no_partial_cm(self, stream, bint debug=False,
                                               int window_size=100,
                                               log_file=None):
        cdef:
            np.ndarray X
            int y, row_i, yhat, weight
            double start_time, elapsed_time, current_acc

        self.last_scores = np.zeros(window_size, dtype=np.float64)
        self.last_pos = 0
        self.FIRST_CYCLE = True

        current_acc = -1
        # learn first example
        row_i, (X, y) = next(stream)

        # start computing time after loading dataset
        start_time = time.time()
        weight = 1
        self._algorithm.train(X, y, weight)
        # learn from other examples
        for row_i, (X, y) in stream:
            yhat = self._algorithm.predict(X)
            self._update_pred_stats(y, yhat)
            self._algorithm.train(X, y, weight)
        elapsed_time = time.time() - start_time
        stats = self.stats
        stats['train_time'] = elapsed_time
        if log_file:
            self._update_log(row_i, elapsed_time, current_acc)
            self._write_log(log_file, window_size)
            self._write_cm_log(log_file)
