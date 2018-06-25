import time
import numpy as np
import sklearn.cluster
from sklearn.cluster import KMeans
from cython.parallel import prange
cimport cython
cimport numpy as np

def make_points(n_sample, n_dim):
    return np.random.random_sample((n_sample, n_dim)).astype('float32')

def print_res(n_sample, n_cluster, n_dim, n_iter, t0, t1, whose):
    gflops = 1.0e-9 * n_sample * n_iter * n_cluster * n_dim / (t1 - t0)
    print('Time ({} KMeans)): {} s'.format(whose, (t1 - t0) / n_iter))
    print('      Performance: {} Glops/s'.format(gflops))
    print('')

##################
# sklearn KMeans #
##################
def kmeans_sklearn_KMeans(points, n_cluster, nb_iter):
    t0 = time.time()
    estimator = KMeans(init='random',
                       n_init=1,
                       tol=1.0e-16,
                       n_clusters=n_cluster,
                       random_state=0,
                       max_iter=nb_iter,
                       algorithm="full",
                       n_jobs=1)
    estimator.fit(points)
    t1 = time.time()
    n_sample, n_dim = points.shape
    n_iter = estimator.n_iter_
    print_res(n_sample, n_cluster, n_dim, n_iter, t0, t1, "sklearn")
  
##############
# ff k-means #
##############
def kmeans_ff_kmeans(points, n_cluster, nb_iter):
    n_sample, n_dim = points.shape
    points_il = points.flatten()
    clusters_il = np.arange(0, n_sample, dtype='int32') % n_cluster
    centroids_il = np.zeros(n_cluster * n_dim, dtype='float32')
    pops_il = np.zeros(n_cluster, dtype='int32')

    cdef:
        float[::1] points_view = points_il
        int[::1] clusters_view = clusters_il
        float[::1] centroids_view = centroids_il
        int[::1] pops_view = pops_il

    t0 = time.time()
    ff_kmeans(points_view, clusters_view, centroids_view, pops_view, n_dim, nb_iter)
    t1 = time.time()
    n_iter = nb_iter
    print_res(n_sample, n_cluster, n_dim, n_iter, t0, t1, "ff inside loop")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ff_kmeans(float[::1] points, int[::1] clusters,
                     float[::1] centroids, int[::1] pops,
                     int n_dim, int nb_iterations) nogil:
    cdef:
        int n_sample = len(points) / n_dim
        int n_cluster = len(centroids) / n_dim
        int sample, cluster, iteration, dim, best_cluster
        float alpha, x, distance, best_distance

    for iteration in range(nb_iterations):
        for cluster in range(n_cluster):
            for dim in range(n_dim):
                centroids[cluster * n_dim + dim] = 0.0
            pops[cluster] = 0
        for sample in range(n_sample):
            cluster = clusters[sample]
            for dim in range(n_dim):
                centroids[cluster * n_dim + dim] += points[sample * n_dim + dim]
            pops[cluster] += 1
        for cluster in range(n_cluster):
            if pops[cluster] > 0:
                alpha = 1.0 / pops[cluster]
                for dim in range(n_dim):
                    centroids[cluster * n_dim + dim] *= alpha

        for sample in prange(n_sample):
            best_distance = n_dim + 1.0
            best_cluster = -1
            for cluster in range(n_cluster):
                distance = 0.0
                for dim in range(n_dim):
                    x = points[sample * n_dim + dim] - centroids[cluster * n_dim + dim]
                    distance = distance + x * x
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster
            clusters[sample] = best_cluster

######################
# full python kmeans #
######################
def kmeans_python_kmeans(points, n_cluster, nb_iter):
    n_sample, n_dim = points.shape
    centroids = points[np.random.choice(n_sample, n_cluster, replace=False)]

    t0 = time.time()
    python_kmeans(points, n_cluster, nb_iter)
    t1 = time.time()
    n_iter = nb_iter
    print_res(n_sample, n_cluster, n_dim, n_iter, t0, t1, "pure python")

def python_kmeans(points, n_cluster, nb_iter):
    n_sample, n_dim = points.shape
    centroids = points[np.random.choice(n_sample, n_cluster, replace=False)]
    for iteration in range(nb_iter):
        p_distances = np.linalg.norm(points.reshape(n_sample, n_dim, -1) - centroids.T.reshape(-1, n_dim, n_cluster), axis=1)
        closests = np.argmin(p_distances, axis=1)
        
        for cluster in range(n_cluster):
            mask = closests == cluster
            centroids[cluster,:] = points[mask,:].mean(axis=0)

#################
# cython kmeans # no memorview
#################
def kmeans_cython_kmeans(points, n_cluster, nb_iter):
    n_sample, n_dim = points.shape
    centroids = points[np.random.choice(n_sample, n_cluster, replace=False)]
    clusters = np.zeros(n_sample, dtype=np.int32)
    pops = np.zeros(n_cluster, dtype=np.int32)

    cdef:
        np.ndarray[np.float32_t, ndim=2] cpoints = points
        np.ndarray[np.int32_t, ndim=1] cclusters = clusters
        np.ndarray[np.float32_t, ndim=2] ccentroids = centroids
        np.ndarray[np.int32_t, ndim=1] cpops = pops

    t0 = time.time()
    cython_kmeans(cpoints, cclusters, ccentroids, cpops, nb_iter)
    t1 = time.time()
    n_iter = nb_iter
    print_res(n_sample, n_cluster, n_dim, n_iter, t0, t1, "no memoryview inside loop")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cython_kmeans(np.ndarray[np.float32_t, ndim=2, mode="c"] points,
                         np.ndarray[np.int32_t, ndim=1, mode="c"] clusters,
                         np.ndarray[np.float32_t, ndim=2, mode="c"] centroids,
                         np.ndarray[np.int32_t, ndim=1, mode="c"] pops,
                         int nb_iter):
    cdef:
        int n_dim = points.shape[1]
        int n_sample = points.shape[0]
        int n_cluster = centroids.shape[0]

    cdef:
        float x, distance, min_distance
        int best_cluster
        int iteration, sample, cluster, dim

    with nogil:
        for iteration in xrange(nb_iter):
            for sample in prange(n_sample):
                min_distance = n_dim + 1.0
                best_cluster = -1
                for cluster in xrange(n_cluster):
                    distance = 0.0
                    for dim in xrange(n_dim):
                        x = points[sample, dim] - centroids[cluster, dim]
                        distance = distance + x * x
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster
                clusters[sample] = best_cluster

            for cluster in xrange(n_cluster):
                for dim in xrange(n_dim):
                    centroids[cluster, dim] = 0.0 
                pops[cluster] = 0
            for sample in xrange(n_sample):
                for dim in xrange(n_dim):
                    centroids[clusters[sample], dim] += points[sample, dim]
                    pops[clusters[sample]] += 1
            for cluster in xrange(n_cluster):
                if pops[cluster] > 0:
                    for dim in xrange(n_dim):
                        centroids[cluster, dim] /= pops[cluster]
