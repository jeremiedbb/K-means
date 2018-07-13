import time
import numpy as np
import sys
import sklearn
from sklearn.cluster import KMeans


# KMeans
def kmeans(points, clusters, n_iter, distrib, algo, tol, init):
    n_sample = points.shape[0]
    n_feature = points.shape[1]
    n_component = clusters.shape[0]
    
    if init != 'no-init':
        pre_init = 'y'
    else:
        clusters = 'k-means++'
        pre_init = 'n'

    t = time.time()
    if distrib == 'sklearn':
        if tol is False:
            km = KMeans(init=clusters,
                        n_init=1,
                        n_clusters=n_component,
                        random_state=0,
                        algorithm=algo,
                        precompute_distances=True)
        else:
            km = KMeans(init=clusters,
                        n_init=1,
                        n_clusters=n_component,
                        random_state=0,
                        algorithm=algo,
                        precompute_distances=True,
                        tol=0,
                        max_iter=n_iter)
    else:
        if tol is False:
            km = KMeans(init=clusters,
                        n_init=1,
                        n_clusters=n_component,
                        random_state=0,
                        algorithm=algo)
        else:
            km = KMeans(init=clusters,
                        n_init=1,
                        n_clusters=n_component,
                        random_state=0,
                        algorithm=algo,
                        tol=0,
                        max_iter=n_iter)
    
    km.fit(points)
    #print(km.inertia_)
    t = time.time() - t

    print('fit,' + distrib + ',' + pre_init + ',' + algo + ',' + str(tol) + ',' + str(n_sample) + ',' + str(n_feature) + ',' + str(n_component) + ',' + str(t) + ',' + str(km.n_iter_))


points = np.loadtxt("points.csv", delimiter=',', dtype=np.float32)
clusters = np.loadtxt("clusters.csv", delimiter=',', dtype=np.float32)

n_iter = int(sys.argv[1])

distrib = sys.argv[2]
if distrib == 'sklearn':
    sklearn.set_config(working_memory=1000)

algo = sys.argv[3]

tol = bool(int(sys.argv[4]))

init = sys.argv[5]

kmeans(points, clusters, n_iter, distrib, algo, tol, init)
