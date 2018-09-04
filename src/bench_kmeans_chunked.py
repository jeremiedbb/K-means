import numpy as np
from kmeans_chunked import kmeans_lloyd_chunked, _inertia
import time


def kmeans(X, centers, n_samples_chunk, max_iter):
    x_squared_norms = (X**2).sum(axis=1)

    centers_old = np.zeros_like(centers)
    centers_new = centers.copy()
    labels = np.zeros(X.shape[0], dtype=np.int32)

    for i in range(max_iter):

        kmeans_lloyd_chunked(X,
                             x_squared_norms,
                             centers_old,
                             centers_new,
                             labels,
                             n_samples_chunk)

        center_shift = ((centers_new - centers_old)**2).sum()
        if center_shift <= 0.0:
            break

    if center_shift > 0:
        kmeans_lloyd_chunked(X,
                             x_squared_norms,
                             centers_old,
                             centers_new,
                             labels,
                             n_samples_chunk)

    inertia = _inertia(X, centers_old, labels)

    return labels, centers_old, inertia, i + 1


n_samples_pow = 12
n_features = 2**1
n_clusters = 2**10
X = np.random.random_sample((2**n_samples_pow, n_features)).astype(np.float32)
centers = X[np.random.choice(np.arange(X.shape[0]), n_clusters)]

n_tests = 20

for n_samples_chunk in [2**(j) for j in np.arange(4, n_samples_pow + 1)]:
    ts = []
    for test in range(n_tests):
        t = time.time()
        labels, centers_old, inertia, n_iter = \
            kmeans(X, centers, n_samples_chunk + 1, 300)
        t = time.time() - t
        t /= n_iter
        ts.append(t)

    tm = sum(ts) / n_tests
    tstd = sum([t*t for t in ts]) / n_tests - tm * tm
    print(str(n_samples_chunk) + ',' + str(tm) + ',' + str(tstd))
