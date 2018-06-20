import time
import numpy as np
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from cython.parallel import prange
cimport cython


cdef int width = 427
cdef int height = 640
cdef int nb_points = width * height
cdef int nb_clusters = 1024
cdef int nb_iterations_insideloop = 100
cdef int nb_iterations_scikitlearn = 3
cdef int nb_iterations_intel = 20

# ~ 270000 3D points
pixels = np.random.rand(width * height, 3).astype('float32')
print("shape :", pixels.shape)
print(pixel.shape)
pr = np.array(pixel[:, 0])
pg = np.array(pixel[:, 1])
pb = np.array(pixel[:, 2])
cluster = np.arange(0, nb_points, dtype = 'int32')
cluster = cluster % nb_clusters
cr = np.zeros(nb_clusters, dtype = 'float32')
cg = np.zeros(nb_clusters, dtype = 'float32')
cb = np.zeros(nb_clusters, dtype = 'float32')
pop = np.zeros(nb_clusters, dtype = 'int32')

cdef float[::1] pr_view = pr
cdef float[::1] pg_view = pg
cdef float[::1] pb_view = pb
cdef int[::1] cluster_view = cluster
cdef float[::1] cr_view = cr
cdef float[::1] cg_view = cg
cdef float[::1] cb_view = cb
cdef int[::1] pop_view = pop

t0 = time.time()
kmeans(pr_view, pg_view, pb_view,
       cluster_view,
       cr_view, cg_view, cb_view,
       pop_view, nb_iterations_insideloop)
t1 = time.time()
gflops = 1.0e-9 * width * height * nb_iterations_insideloop * nb_clusters * 8 / (t1 - t0)
print('  Time for KMeans clustering, InsideLoop: {} s'.format((t1 - t0) / nb_iterations_insideloop))
print('                             Performance: {} Glops/s'.format(gflops))

t0 = time.time()
res = sklearn.cluster.k_means(pixel, nb_clusters, init = 'random',
    n_init = 1, tol = 1.0e-16, max_iter = nb_iterations_scikitlearn, return_n_iter = True)
t1 = time.time()
gflops = 1.0e-9 * width * height * nb_iterations_scikitlearn * nb_clusters * 8 / (t1 - t0)
print('Time for KMeans clustering, Scikit-Learn: {} s'.format((t1 - t0) / nb_iterations_scikitlearn))
print('                             Performance: {} Glops/s'.format(gflops))

t0 = time.time()
res = KMeans(init = 'random', n_init = 1, tol = 1.0e-16, n_clusters = nb_clusters,
    random_state = 0, max_iter = nb_iterations_intel).fit(pixel)
t1 = time.time()
gflops = 1.0e-9 * width * height * nb_iterations_intel * nb_clusters * 8 / (t1 - t0)
print('       Time for KMeans clustering, Intel: {} s'.format((t1 - t0) / nb_iterations_scikitlearn))
print('                             Performance: {} Glops/s'.format(gflops))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void kmeans(float[::1] pr, float[::1] pg, float[::1] pb,
                  int[::1] cluster,
                  float[::1] cr, float[::1] cg, float[::1] cb,
                  int[::1] pop, int nb_iterations) nogil:
  cdef int m = len(cluster) # m is the number of points
  cdef int n = len(pop)     # n is the number of clusters
  cdef int i, j, k, best_j
  cdef float alpha, distance, best_distance
  cdef float x, y, z

  for k in range(nb_iterations):
    for j in range(n):
      cr[j] = 0.0
      cg[j] = 0.0
      cb[j] = 0.0
      pop[j] = 0
    for i in range(m):
      j = cluster[i]
      cr[j] += pr[i]
      cg[j] += pg[i]
      cb[j] += pb[i]
      pop[j] += 1
    for j in range(n):
      if pop[j] > 0:
        alpha = 1.0 / pop[j]
        cr[j] *= alpha
        cg[j] *= alpha
        cb[j] *= alpha

    for i in prange(m):
      best_distance = 3.0 + 1.0
      best_j = 0
      for j in range(n):
        x = pr[i] - cr[j]
        y = pg[i] - cg[j]
        z = pb[i] - cb[j]
        distance = x * x + y * y + z * z
        if distance < best_distance:
          best_distance = distance
          best_j = j
      cluster[i] = best_j