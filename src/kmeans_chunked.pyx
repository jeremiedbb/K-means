#cython: profile=True, initializedcheck=False, nonecheck=False, boundscheck=False, wraparound=False, cdivision=True, overflowcheck=False, overflowcheck.fold=False
cimport numpy as np
import numpy as np
cimport cython
cimport openmp
from cython cimport floating
from cython.parallel cimport prange, parallel
from scipy.linalg.cython_blas cimport sgemm, dgemm
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy


cdef void _gemm(char *ta, char *tb, int *m, int *n, int *k,
                floating *alpha, floating *A, int *lda, floating *B,
                int *ldb, floating *beta, floating *C, int *ldc) nogil:
    if floating is float:
        sgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    else:
        dgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef void _labels_inertia_centers_chunk(floating *X_chunk,
                                        floating *sample_weight_chunk,
                                        floating *x_squared_norms_chunk,
                                        floating *centers_old,
                                        floating *centers_new,
                                        floating *centers_squared_norms,
                                        floating *weight_in_cluster,
                                        floating *pairwise_distances,
                                        int *labels_chunk,
                                        int n_samples_chunk,
                                        int n_clusters,
                                        int n_features,
                                        floating *inertia) nogil:
    """K-means combined EM step for one data chunk
    
    Compute the partial contribution of a single data chunk to the labels,
    centers and inertia.
    """
    cdef:   
        floating sq_dist_eff, min_sq_dist_eff, weight
        int i, j, k, best_cluster
    
        floating alpha = -2.0
        floating beta = 1.0
        char *trans_data = 'n'
        char *trans_centers = 't'

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers C.
    for i in xrange(n_samples_chunk):
        for j in xrange(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]
    
    _gemm(trans_centers, trans_data, &n_clusters, &n_samples_chunk,
          &n_features, &alpha, centers_old, &n_features, X_chunk, &n_features,
          &beta, pairwise_distances, &n_clusters)

    for i in xrange(n_samples_chunk):
        min_sq_dist_eff = pairwise_distances[i * n_clusters]
        best_cluster = 0
        for j in xrange(n_clusters):
            sq_dist_eff = pairwise_distances[i * n_clusters + j]
            if min_sq_dist_eff > sq_dist_eff:
                min_sq_dist_eff = sq_dist_eff
                best_cluster = j

        weight = sample_weight_chunk[i]
        if x_squared_norms_chunk[i] + min_sq_dist_eff > 0:
            inertia[0] += weight * (x_squared_norms_chunk[i] + min_sq_dist_eff)
        labels_chunk[i] = best_cluster
        weight_in_cluster[best_cluster] += weight
        for k in xrange(n_features):      
            centers_new[best_cluster * n_features + k] += \
                X_chunk[i * n_features + k] * weight


cpdef floating kmeans_lloyd_chunked(floating[:, ::1] X,
                                    floating[::1] sample_weight,
                                    floating[::1] x_squared_norms,
                                    floating[:, ::1] centers_old,
                                    floating[:, ::1] centers_new,
                                    int[::1] labels):
    """Single interation of K-means lloyd algorithm

    Update labels and centers (inplace), and compute inertia, for one
    iteration, parallelized over data chunks.

    Parameters
    ----------
    X : memoryview, shape (n_samples, n_features)

    x_squared_norms : memoryview, shape (n_samples,)
        Squared L2 norm of X.

    sample_weight : memoryview, shape (n_samples,)
        The weights for each observation in X.

    centers_old : memoryview, shape (n_clusters, n_features)
        Centers before previous iteration.

    centers_new : memoryview, shape (n_clusters, n_features)
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    labels : memoryview, shape (n_samples,)
        labels assignment.

    #n_samples_chunk : int

    Returns
    -------
    inertia : float
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_new.shape[0]

        int n_samples_chunk = 256
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_r = n_samples % n_samples_chunk
        int chunk_idx, n_samples_chunk_eff

        int j, k
        floating alpha

        floating *centers_squared_norms = \
            <floating*> malloc(n_clusters * sizeof(floating))

        floating *weight_in_cluster = \
            <floating*> malloc(n_clusters * sizeof(floating))

        floating inertia = 0.0

        floating *inertia_chunk
        floating *centers_new_chunk
        floating *weight_in_cluster_chunk
        floating *pairwise_distances_chunk

    # count remainder for total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk
    
    # reset all arrays at each iteration
    memcpy(&centers_old[0, 0], &centers_new[0, 0],
           n_clusters * n_features * sizeof(floating))

    memset(centers_squared_norms, 0, n_clusters * sizeof(floating))
    for j in xrange(n_clusters):
        for k in xrange(n_features):
            centers_squared_norms[j] += centers_new[j, k] * centers_new[j, k]

    #memset(&labels[0], 0, n_samples * sizeof(int))
    memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
    memset(weight_in_cluster, 0, n_clusters * sizeof(floating))

    for j in xrange(n_clusters):
        if weight_in_cluster[j] > 0:
            print(weight_in_cluster[j])

    with nogil, parallel():
        inertia_chunk = <floating*> malloc(sizeof(floating))

        centers_new_chunk = \
            <floating*> malloc(n_clusters * n_features * sizeof(floating))

        weight_in_cluster_chunk = \
            <floating*> malloc(n_clusters * sizeof(floating))

        pairwise_distances_chunk = \
            <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))

        inertia_chunk[0] = 0.0
        memset(centers_new_chunk, 0, n_clusters * n_features * sizeof(floating))
        memset(weight_in_cluster_chunk, 0, n_clusters * sizeof(floating))

        for chunk_idx in prange(n_chunks):
            if n_samples_r > 0 and chunk_idx == n_chunks - 1:
                n_samples_chunk_eff = n_samples_r
            else:
                n_samples_chunk_eff = n_samples_chunk

            _labels_inertia_centers_chunk(
                &X[chunk_idx * n_samples_chunk, 0],
                &sample_weight[chunk_idx * n_samples_chunk],
                &x_squared_norms[chunk_idx * n_samples_chunk],
                &centers_old[0, 0],
                centers_new_chunk,
                centers_squared_norms,
                weight_in_cluster_chunk,
                pairwise_distances_chunk,
                &labels[chunk_idx * n_samples_chunk],
                n_samples_chunk_eff,
                n_clusters,
                n_features,
                inertia_chunk)

        # reduce
        with gil:
            (&inertia)[0] += inertia_chunk[0]

            for j in xrange(n_clusters):
                weight_in_cluster[j] += weight_in_cluster_chunk[j]
                for k in xrange(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]

        free(inertia_chunk)
        free(weight_in_cluster_chunk)
        free(centers_new_chunk)
        free(pairwise_distances_chunk)

    # relocate empty clusters
    #
    #
    #

    # average new centers wrt sample weights
    for j in xrange(n_clusters):
        if weight_in_cluster[j] > 0:
            alpha = 1.0 / weight_in_cluster[j]
            for k in xrange(n_features):
                centers_new[j, k] *= alpha

    free(centers_squared_norms)
    free(weight_in_cluster)

    return inertia
