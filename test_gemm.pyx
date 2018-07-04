import time
import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np
from scipy.linalg.cython_blas cimport sgemm


# paiwise dist sgemm then min dist
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.int64_t gemm():
    pts = np.array([[1,0],
                    [0,1],
                    [0,1],
                    [1,0],
                    [1,0]], dtype=np.float32)

    cds = np.array([[1,0],
                    [0,1],
                    [0,1]], dtype=np.float32)

    cdef:
        int n_sample = 5
        int n_feature = 2
        int n_component = 3

    cdef:
        np.int64_t count = 0

    cdef:
        np.ndarray[np.float32_t, ndim=2] points = pts
        np.ndarray[np.float32_t, ndim=2] centroids = cds
        np.ndarray[np.float32_t, ndim=2] res = np.empty((n_sample, n_component), dtype=np.float32)

    print(points)
    print(centroids)
    
    cdef:
        np.float32_t alpha = 1.0
        np.float32_t beta = 0.0
        char *transa = 'n'
        char *transb = 't'
        np.float32_t *a0=&points[0,0]
        np.float32_t *b0=&centroids[0,0]
        np.float32_t *c0=&res[0,0]
        
    with nogil:
        sgemm(transb, transa, &n_component, &n_sample, &n_feature, &alpha, b0, &n_feature, a0, &n_feature, &beta, c0, &n_component)

    print(res)
    count = n_sample * n_component * n_feature * 2
    return count
