import numpy as np
import sys
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms 

n_sample = int(sys.argv[1])
n_dim = int(sys.argv[2])
n_component = int(sys.argv[3])

points = np.random.random_sample((n_sample, n_dim)).astype('float32')
np.savetxt("points.csv", points, delimiter=",")

x_squared_norms = row_norms(points, squared=True)
clusters = _k_init(points, n_component, x_squared_norms, random_state=np.random.RandomState(0))
np.savetxt("clusters.csv", clusters, delimiter=",")