import math
from sys import getsizeof
from collections import deque


cdef class AdWinList:
    cdef object _nodes
    cdef int _count, _max_buckets

    cpdef double memory_size(self)

    cpdef add_head(self)

    cpdef remove_head(self)

    cpdef add_tail(self)

    cpdef remove_tail(self)

cdef class AdWinListNode:
    cdef int _max_buckets
    cdef int _size
    cdef AdWinListNode _next, _prev
    cdef list _sum, _variance

    cpdef double memory_size(self)

    cpdef add_back(self, double value, double var)

    cpdef drop_front(self, int n=*)

cdef class ADWIN:
    cdef AdWinList _buckets
    cdef int _min_long_wind, _MAXBUCKETS, _last_bucket, _bucket_n
    cdef double _delta, _mint_time, _mint_clock,_mdbl_error
    cdef double _mdbl_width, _sum, _W, _var

    cpdef double memory_size(self)

    cpdef double get_estimation(self)

    cpdef update(self, double value)

    cpdef print_info(self)

    cpdef insert_element(self, double value)

    cpdef compress_buckets(self)

    cpdef bint drop_check_drift(self)

    cpdef delete_element(self)

    cpdef double cut_expression(self, double n0, double n1,
                                double u0, double u1)

    cpdef int bucket_size(self, double row)
