#cython: initializedcheck=False, nonecheck=False, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np

ctypedef fused floating_t:
    np.float32_t
    np.float64_t

cdef void kmeans_chunk(floating_t[:, ::1] X_chunk,
                       floating_t[:, ::1] C,
                       floating_t[:, ::1] sums,
                       np.int32_t[::1] labels,
                       np.int32_t[::1] pops,
                       Py_ssize_t start) nogil:
    cdef:
        Py_ssize_t n_features = X_chunk.shape[1]
        Py_ssize_t n_samples = X_chunk.shape[0]
        Py_ssize_t n_clusters = C.shape[0]

        floating_t x = 0.0
        floating_t sq_dist = 0.0
        floating_t min_sq_dist = -1.0
        Py_ssize_t best_cluster = -1
        Py_ssize_t s_idx = 0
        Py_ssize_t c_idx = 0
        Py_ssize_t f_idx = 0
       
    # pairwise dist argmin min
    for s_idx in xrange(n_samples):
        min_sq_dist = -1.
        best_cluster = -1
        for c_idx in xrange(n_clusters):
            sq_dist = 0.0
            for f_idx in xrange(n_features):
                x = X_chunk[s_idx, f_idx] - C[c_idx, f_idx]
                sq_dist = sq_dist + x * x
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                best_cluster = c_idx
            elif min_sq_dist == -1.0:
                min_sq_dist = sq_dist
                best_cluster = c_idx
        labels[start + s_idx] = best_cluster
    
    # update pops & sums
    for s_idx in xrange(n_samples):
        c_idx = labels[start + s_idx]
        pops[c_idx] += 1
        for f_idx in xrange(n_features):
            sums[c_idx, f_idx] = sums[c_idx, f_idx] + X_chunk[s_idx, f_idx]


def kmeans(floating_t[:, ::1] X,
           floating_t[:, ::1] C,
           Py_ssize_t chunk_n_rows,
           Py_ssize_t n_iter,
           type dtype):
    cdef:
        Py_ssize_t n_features = X.shape[1]
        Py_ssize_t n_samples = X.shape[0]
        Py_ssize_t n_clusters = C.shape[0]
        
        Py_ssize_t chunk_idx = 0
        Py_ssize_t n_chunks = 0
        Py_ssize_t n_samples_chunk = 0
        Py_ssize_t n_samples_r = 0

        Py_ssize_t s_idx = 0
        Py_ssize_t c_idx = 0
        Py_ssize_t f_idx = 0

        Py_ssize_t iter_idx = 0

        np.int32_t[::1] pops = np.zeros(n_clusters, dtype=np.int32)
        np.int32_t[::1] labels = np.zeros(n_samples, dtype=np.int32)
        floating_t[:, ::1] sums = np.zeros((n_clusters, n_features), dtype=dtype)

        floating_t alpha = 0.0
        floating_t inertia = 0.0
        floating_t x = 0.0

    # make chunks
    # 4 = cache_size = chunk size = (chunk_n_rows + n_clusters) * n_features * 4 * 1e-6
    #n_samples_chunk = 1000000 / n_features - n_clusters
    n_samples_chunk = chunk_n_rows

    n_chunks = n_samples // n_samples_chunk
    n_samples_r = n_samples % n_samples_chunk

    with nogil:
        # loop over chunks
        for chunk_idx in prange(n_chunks):
            kmeans_chunk(X[chunk_idx * n_samples_chunk: (chunk_idx + 1)* n_samples_chunk],
                         C, sums, labels, pops, chunk_idx * n_samples_chunk)

        if n_samples_r > 0:
            kmeans_chunk(X[n_chunks * n_samples_chunk: n_chunks * n_samples_chunk + n_samples_r],
                         C, sums, labels, pops, n_chunks * n_samples_chunk)

        # average sums
        for c_idx in xrange(n_clusters):
            if pops[c_idx] != 0:
                alpha = 1.0 / pops[c_idx]
                for f_idx in xrange(n_features):
                    sums[c_idx, f_idx] *= alpha

        # compute new inertia
        for s_idx in xrange(n_samples):
            c_idx = labels[s_idx]
            for f_idx in xrange(n_features):
                x = X[s_idx, f_idx] - sums[c_idx, f_idx]
                inertia += x * x

    return labels, sums, inertia
