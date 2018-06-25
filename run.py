import sys
import misc_kmeans as mk

n_sample = int(sys.argv[1])
n_dim = int(sys.argv[2])
n_cluster = int(sys.argv[3])
n_iter = int(sys.argv[4])
func = int(sys.argv[5])

points = mk.make_points(n_sample, n_dim)

if func == 1:
    kmeans = mk.kmeans_sklearn_KMeans
elif func == 2:
    kmeans = mk.kmeans_ff_kmeans
elif func == 3:
    kmeans = mk.kmeans_python_kmeans
elif func == 4:
    kmeans = mk.kmeans_cython_kmeans

kmeans(points, n_cluster, n_iter)

