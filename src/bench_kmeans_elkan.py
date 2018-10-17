import numpy as np
from sklearn.cluster import KMeans
import time


def bench_one(n_samples, n_clusters, n_features, n_tests):
    X = np.random.random_sample((n_samples, n_features)).astype(np.float32)
    centers = X[np.random.choice(np.arange(X.shape[0]), n_clusters)]

    ts = np.zeros(n_tests)
    for test in range(n_tests):
        t = time.time()

        km = KMeans(n_clusters=n_clusters,
                    init=centers, n_init=1,
                    max_iter=20, tol=0,
                    algorithm='elkan')
        km.fit(X)

        t = time.time() - t
        t /= km.n_iter_
        ts[test] = t

    tm = ts.mean()
    tstd = ts.std()
    print(str(tm) + ',' + str(tstd))


n_tests = 5
n_samples = 10000
n_clusters = 1000

for n_features in [2**j for j in np.arange(1, 15)]:
    bench_one(n_samples, n_clusters, n_features, n_tests)
