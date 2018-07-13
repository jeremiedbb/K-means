import time
import numpy as np
import sys
import sklearn
from sklearn.cluster import KMeans

# KMeans
def kmeans(points, clusters, n_iter, precompute):
    n_sample = points.shape[0]
    n_feature = points.shape[1]
    n_component = clusters.shape[0]
    
    t = time.time()
    km = KMeans(init=clusters,
                n_init=1,
                tol=1.0e-16,
                n_clusters=n_component,
                random_state=0,
                max_iter=n_iter,
                precompute_distances=precompute,
                algorithm='full')
    km.fit(points)
    t = time.time() - t

    print(str(n_sample) + ',' + str(n_feature) + ',' + str(n_component) + ',' + str(t) + ',' + str(km.n_iter_))


points = np.loadtxt("points.csv", delimiter=',', dtype=np.float32)
clusters = np.loadtxt("clusters.csv", delimiter=',', dtype=np.float32)

n_iter = int(sys.argv[1])

distrib = sys.argv[2]
if distrib == '-sklearn':
    sklearn.set_config(working_memory=24)

precompute = bool(sys.argv[3])

kmeans(points, clusters, n_iter, precompute)