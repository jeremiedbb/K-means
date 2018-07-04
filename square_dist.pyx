import time
import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from scipy.linalg.cython_blas cimport sgemm

def print_res(ops, t, n_iter, func_name, option='full'):
    gflops = 1.0e-9 * ops / t
    if option == 'full':
        print('     Function: {}'.format(func_name))
        print('    Time/iter: {} s'.format(t / n_iter))
        print(' Num ops/iter: {}'.format(ops / n_iter))
        print('  Performance: {} Gflops/s'.format(gflops))
    else:
        print(gflops)

def square_dist(func, points, n_component, n_iter):
    n_sample, n_feature = points.shape
    centroids = points[np.random.choice(n_sample, n_component, replace=False)]
    clusters = np.zeros(n_sample, dtype=np.int32)

    ops, t = func(points, clusters, centroids, n_iter)

    return ops, t
    #print_res(ops, t, n_iter, func.__name__, option='')


# full loop min dist inside no prange
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double full_loop_min_dist_no_prange(np.ndarray[np.float32_t, ndim=2] points,
                                          np.ndarray[np.int32_t, ndim=1] clusters,
                                          np.ndarray[np.float32_t, ndim=2] centroids,
                                          int n_iter):
    cdef:
        int n_sample = points.shape[0]
        int n_feature = points.shape[1]
        int n_component = centroids.shape[0]

    cdef:
        int iter_idx, sample_idx, feature_idx, component_idx
        float x, square_dist, min_square_dist
        int best_component_idx
        np.float64_t count = 1.0
        
    with nogil:
        for iter_idx in xrange(n_iter):
            for sample_idx in xrange(n_sample):
                min_square_dist = n_feature + 1.0
                best_component_idx = -1
                for component_idx in xrange(n_component):
                    square_dist = 0.0
                    for feature_idx in xrange(n_feature):
                        x = points[sample_idx, feature_idx] - centroids[component_idx, feature_idx]
                        square_dist = square_dist + x * x
                    if square_dist < min_square_dist:
                        min_square_dist = square_dist
                        best_component_idx = component_idx
                clusters[sample_idx] = best_component_idx

        count *= n_sample
        count *= n_component
        count *= n_feature
        count *= n_iter
        count *= 3
        return count

# full loop min dist inside w/ prange sample
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double full_loop_min_dist_prange_sample(np.ndarray[np.float32_t, ndim=2] points,
                                              np.ndarray[np.int32_t, ndim=1] clusters,
                                              np.ndarray[np.float32_t, ndim=2] centroids,
                                              int n_iter):
    cdef:
        int n_sample = points.shape[0]
        int n_feature = points.shape[1]
        int n_component = centroids.shape[0]

    cdef:
        int iter_idx, sample_idx, feature_idx, component_idx
        float x, square_dist, min_square_dist
        int best_component_idx
        np.float64_t count = 1.0
        
    with nogil:
        for iter_idx in xrange(n_iter):
            for sample_idx in prange(n_sample):
                min_square_dist = n_feature + 1.0
                best_component_idx = -1
                for component_idx in xrange(n_component):
                    square_dist = 0.0
                    for feature_idx in xrange(n_feature):
                        x = points[sample_idx, feature_idx] - centroids[component_idx, feature_idx]
                        square_dist = square_dist + x * x
                    if square_dist < min_square_dist:
                        min_square_dist = square_dist
                        best_component_idx = component_idx
                clusters[sample_idx] = best_component_idx

        count *= n_sample
        count *= n_component
        count *= n_feature
        count *= n_iter
        count *= 3
        return count

# sklearn pairwise dist + argmin
def sklearn_pairwise_dist_argmin(points, clusters, centroids, n_iter):
    n_sample = points.shape[0]
    n_feature = points.shape[1]
    n_component = centroids.shape[0]
    
    t = time.time()
    for iter_idx in range(n_iter):
        clusters, min_dists = pairwise_distances_argmin_min(points, centroids,
                                                            metric_kwargs={"squared":True})
    t = time.time() - t

    return n_sample * n_feature * n_component * n_iter * 2, t

# paiwise dist full loop then min dist
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double pairwise_dist_full_loop_min_dist(np.ndarray[np.float32_t, ndim=2] points,
                                              np.ndarray[np.int32_t, ndim=1] clusters,
                                              np.ndarray[np.float32_t, ndim=2] centroids,
                                              int n_iter):
    cdef:
        int n_sample = points.shape[0]
        int n_feature = points.shape[1]
        int n_component = centroids.shape[0]

    cdef:
        int iter_idx, sample_idx, feature_idx, component_idx
        float x, square_dist, min_square_dist
        int best_component_idx
        np.float64_t count = 1.0

    cdef:
        np.ndarray[np.float32_t, ndim=1] points_snorms = np.zeros(n_sample, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] centroids_snorms = np.zeros(n_component, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] dots = np.zeros((n_sample, n_component), dtype=np.float32)
        
    with nogil:
        # points squared norms
        for sample_idx in prange(n_sample):
            for feature_idx in xrange(n_feature):
                x = points[sample_idx, feature_idx]
                points_snorms[sample_idx] = points_snorms[sample_idx] + x * x
        
        for iter_idx in xrange(n_iter):
            # centroids squared norms
            for component_idx in prange(n_component):
                for feature_idx in xrange(n_feature):
                    x = centroids[component_idx, feature_idx]
                    centroids_snorms[component_idx] = centroids_snorms[component_idx] + x * x

            # points * centroids
            for sample_idx in prange(n_sample):
                for component_idx in xrange(n_component):
                    for feature_idx in xrange(n_feature):
                        dots[sample_idx, component_idx] = dots[sample_idx, component_idx] + points[sample_idx, feature_idx] * centroids[component_idx, feature_idx]

            for sample_idx in prange(n_sample):
                min_square_dist = n_feature + 1.0
                best_component_idx = -1
                for component_idx in xrange(n_component):
                    square_dist = square_dist + points_snorms[sample_idx] + centroids_snorms[component_idx] - 2 * dots[sample_idx, component_idx]
                    if square_dist < min_square_dist:
                        min_square_dist = square_dist
                        best_component_idx = component_idx
                clusters[sample_idx] = best_component_idx

        count *= n_sample
        count *= n_component
        count *= n_feature
        count *= n_iter
        count *= 2
        return count

# paiwise dist sgemm then min dist
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pairwise_dist_sgemm_min_dist(np.ndarray[np.float32_t, ndim=2] points,
                                              np.ndarray[np.int32_t, ndim=1] clusters,
                                              np.ndarray[np.float32_t, ndim=2] centroids,
                                              int n_iter):
    cdef:
        int n_sample = points.shape[0]
        int n_feature = points.shape[1]
        int n_component = centroids.shape[0]

        int iter_idx, sample_idx, feature_idx, component_idx
        float x, square_dist, min_square_dist
        int best_component_idx
        double count = 1.0
        np.float64_t t = 0.0

        np.ndarray[np.float32_t, ndim=1] points_snorms = np.zeros(n_sample, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] centroids_snorms = np.zeros(n_component, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] dots = np.zeros((n_sample, n_component), dtype=np.float32)
    
        np.float32_t alpha = 1.0
        np.float32_t beta = 0.0
        char *transa = 'n'
        char *transb = 't'
        np.float32_t *a0=&points[0,0]
        np.float32_t *b0=&centroids[0,0]
        np.float32_t *c0=&dots[0,0]
        
    t = time.time()
    with nogil:
        for sample_idx in xrange(n_sample):
            for feature_idx in xrange(n_feature):
                x = points[sample_idx, feature_idx]
                points_snorms[sample_idx] = points_snorms[sample_idx] + x * x
        
        for iter_idx in xrange(n_iter):
            for component_idx in xrange(n_component):
                for feature_idx in xrange(n_feature):
                    x = centroids[component_idx, feature_idx]
                    centroids_snorms[component_idx] = centroids_snorms[component_idx] + x * x

            sgemm(transb, transa, &n_component, &n_sample, &n_feature, &alpha, b0, &n_feature, a0, &n_feature, &beta, c0, &n_component)

            for sample_idx in prange(n_sample):
                min_square_dist = n_feature + 1.0
                best_component_idx = -1
                for component_idx in xrange(n_component):
                    square_dist = square_dist + points_snorms[sample_idx] + centroids_snorms[component_idx] - 2 * dots[sample_idx, component_idx]
                    if square_dist < min_square_dist:
                        min_square_dist = square_dist
                        best_component_idx = component_idx
                clusters[sample_idx] = best_component_idx
    t = time.time() - t

    count *= n_sample
    count *= n_component 
    count *= n_feature 
    count *= n_iter 
    count *= 2
    return count, t

# paiwise dist matmul then min dist
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pairwise_dist_matmul_min_dist(np.ndarray[np.float32_t, ndim=2] points,
                                               np.ndarray[np.int32_t, ndim=1] clusters,
                                               np.ndarray[np.float32_t, ndim=2] centroids,
                                               int n_iter):
    cdef:
        int n_sample = points.shape[0]
        int n_feature = points.shape[1]
        int n_component = centroids.shape[0]

        int iter_idx, sample_idx, feature_idx, component_idx
        float x, square_dist, min_square_dist
        int best_component_idx
        double count = 1.0
        np.float64_t t = 0.0

        np.ndarray[np.float32_t, ndim=1] points_snorms = np.zeros(n_sample, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] centroids_snorms = np.zeros(n_component, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] dots = np.zeros((n_sample, n_component), dtype=np.float32)
        
    t = time.time()
    with nogil:
        for sample_idx in xrange(n_sample):
            for feature_idx in xrange(n_feature):
                x = points[sample_idx, feature_idx]
                points_snorms[sample_idx] = points_snorms[sample_idx] + x * x
        
        for iter_idx in xrange(n_iter):
            for component_idx in xrange(n_component):
                for feature_idx in xrange(n_feature):
                    x = centroids[component_idx, feature_idx]
                    centroids_snorms[component_idx] = centroids_snorms[component_idx] + x * x

            with gil:
                np.matmul(points, np.transpose(centroids), out=dots)

            for sample_idx in prange(n_sample):
                min_square_dist = n_feature + 1.0
                best_component_idx = -1
                for component_idx in xrange(n_component):
                    square_dist = square_dist + points_snorms[sample_idx] + centroids_snorms[component_idx] - 2 * dots[sample_idx, component_idx]
                    if square_dist < min_square_dist:
                        min_square_dist = square_dist
                        best_component_idx = component_idx
                clusters[sample_idx] = best_component_idx
    t = time.time() - t

    count *= n_sample
    count *= n_component 
    count *= n_feature 
    count *= n_iter 
    count *= 2
    return count, t