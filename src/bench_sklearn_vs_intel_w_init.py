import time
import numpy as np
import sys
import sklearn
from sklearn.cluster import KMeans

# KMeans
def kmeans(points, n_component, n_iter, distrib):
    n_sample = points.shape[0]
    n_feature = points.shape[1]
    
    t = time.time()
    if distrib == 'sklearn':
        km = KMeans(init='k-means++',
                    n_init=1,
                    tol=0.0,
                    n_clusters=n_component,
                    random_state=0,
                    max_iter=n_iter,
                    algorithm='full',
                    precompute_distances=True)
    else:
        km = KMeans(init='k-means++',
                    n_init=1,
                    tol=0.0,
                    n_clusters=n_component,
                    random_state=0,
                    max_iter=n_iter,
                    algorithm='full')
    km.fit(points)
    print(km.inertia_)
    t = time.time() - t

    print('fit,' + distrib + ',' + str(n_sample) + ',' + str(n_feature) + ',' + str(n_component) + ',' + str(t) + ',' + str(km.n_iter_))


points = np.loadtxt("points.csv", delimiter=',', dtype=np.float32)

n_iter = int(sys.argv[1])

distrib = sys.argv[2]
if distrib == 'sklearn':
    sklearn.set_config(working_memory=1000)

n_component = int(sys.argv[3])

kmeans(points, n_component, n_iter, distrib)

