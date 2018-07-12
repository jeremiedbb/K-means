#cython: initializedcheck=False, nonecheck=False, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free
cimport cython
cimport numpy as np
import time

def ploop(N):
    a = np.zeros(N).astype(np.float32)
    t = time.time()
    for i in range (1000):
        loop(a)
    t = time.time() - t
    return(t/1000)

def ploop_c(N, n):
    a = np.zeros(N).astype(np.float32)
    cdef np.float32_t[::1] av = a
    t = time.time()
    for i in range (1000):
        chunked_loop(av, n)
    t = time.time() - t
    return(t/1000)

def ploop_m_nc(N):
    a = np.zeros(N).astype(np.float32)
    cdef np.float32_t[::1] av = a
    t = time.time()
    for i in range (1000):
        loop_m_nc(av)
    t = time.time() - t
    return(t/1000)

cpdef void loop(np.ndarray[np.float32_t] a):
    cdef:
        np.float32_t c = 1.0
        Py_ssize_t idx, N = a.shape[0]
    
    with nogil:
        for idx in prange(N):
            a[idx] = 0.0
        for idx in prange(N):
            a[idx] = a[idx] + c

cpdef double chunked_loop(np.float32_t[::1] a, int chunk_size) nogil:
    cdef:
        np.float32_t c = 1.0
        Py_ssize_t expidx, chidx, s, idx, N = a.shape[0]
        Py_ssize_t ch_s = chunk_size
        Py_ssize_t Nch = N / ch_s
        Py_ssize_t rch = N % ch_s

    with gil:
        t = time.time()
    for expidx in range(10000):
        s = 0
        for chidx in range(Nch):
            for idx in range(ch_s):
                a[s + idx] = 0.0
            for idx in range(ch_s):
                a[s + idx] = a[s + idx] + c
            s += chunk_size
        for idx in range(rch):
            a[s + idx] = 0.0
        for idx in range(rch):
            a[s + idx] = a[s + idx] + c
    with gil:
        t = time.time() - t
        return t

cpdef void loop_m_nc(np.float32_t[::1] a) nogil:
    cdef:
        np.float32_t c = 1.0
        Py_ssize_t idx, N = a.shape[0]

    for idx in range(N):
        a[idx] = 0.0
    for idx in range(N):
        a[idx] = a[idx] + c
