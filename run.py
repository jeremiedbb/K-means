import sys
import misc_kmeans as mk
import numpy as np

n_cluster = int(sys.argv[1])
n_iter = int(sys.argv[2])
func = int(sys.argv[3])

points = np.loadtxt("points.csv", delimiter=',', dtype=np.float32)

if func == 1:
    kmeans = mk.kmeans_sklearn_KMeans
elif func == 2:
    kmeans = mk.kmeans_ff_kmeans
elif func == 3:
    kmeans = mk.kmeans_python_kmeans
elif func == 4:
    kmeans = mk.kmeans_cython_kmeans
elif func == 5:
    kmeans = mk.kmeans_cython_flat_kmeans
elif func == 6:
    kmeans = mk.kmeans_cython_des_kmeans
elif func == 7:
    kmeans = mk.kmeans_cython_mm_kmeans

kmeans(points, n_cluster, n_iter)

