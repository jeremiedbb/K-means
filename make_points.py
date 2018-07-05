import numpy as np
import sys
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms 
import time

n_sample = int(sys.argv[1])
n_dim = int(sys.argv[2])
n_component = int(sys.argv[3])

points = np.random.random_sample((n_sample, n_dim)).astype('float32')
np.savetxt("points.csv", points, delimiter=",")

init = sys.argv[4]
if init != 'no-init':
    x_squared_norms = row_norms(points, squared=True)
    t = time.time()
    #clusters = _k_init(points, n_component, x_squared_norms, random_state=np.random.RandomState(0))
    clusters = points[np.random.choice(n_sample, n_component, replace=False)]
    t = time.time() - t
    np.savetxt("clusters.csv", clusters, delimiter=",")

    print('init,,' + str(n_sample) + ',' + str(n_dim) + ',' + str(n_component) + ',' + str(t) + ',O')