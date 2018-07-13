#cython: initializedcheck=False, nonecheck=False, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np
import time

cpdef double chunked_loop(np.float32_t[::1] a, int chunk_size) nogil:
    cdef:
        np.float32_t c = 1.0
        Py_ssize_t exp_idx, chunk_idx, start_idx, idx, N = a.shape[0]
        Py_ssize_t chunksize = chunk_size
        Py_ssize_t N_chunks = N / chunksize
        Py_ssize_t remainder = N % chunksize

    with gil:
        t = time.time()
    for exp_idx in range(1000):
        start_idx = 0
        for chunk_idx in range(N_chunks):
            for idx in range(chunksize):
                a[start_idx + idx] = c
            start_idx += chunksize
        for idx in range(remainder):
            a[start_idx + idx] = c
    with gil:
        t = time.time() - t
        t /= 1000
        return t

cpdef double chunked_loop_b(np.ndarray[np.float32_t] a, int chunk_size):
    cdef:
        np.float32_t c = 1.0
        Py_ssize_t exp_idx, chunk_idx, start_idx, idx, N = a.shape[0]
        Py_ssize_t chunksize = chunk_size
        Py_ssize_t N_chunks = N / chunksize
        Py_ssize_t remainder = N % chunksize

    t = time.time()
    with nogil:
        for exp_idx in range(1000):
            start_idx = 0
            for chunk_idx in range(N_chunks):
                for idx in range(chunksize):
                    a[start_idx + idx] = c
                start_idx += chunksize
            for idx in range(remainder):
                a[start_idx + idx] = c
    t = time.time() - t
    t /= 1000
    return t
