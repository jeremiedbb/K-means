from scipy.sparse import random
from sklearn.cluster import KMeans
import time
import numpy as np


n_samples = [2**i for i in np.arange(11, 17)]
n_clusters = [2**j for j in np.arange(1, 11)]
n_features = [2**k for k in np.arange(6, 11)]


def bench(dtype, density, n_tests, impl):
    n_tot = len(n_samples) * len(n_clusters) * len(n_features)
    ts = np.zeros((n_tests, n_tot))

    for test in range(n_tests):
        idx = 0
        for ns in n_samples:
            for nc in n_clusters:
                for nf in n_features:
                    X = random(ns, nf, density=density,
                               format='csr', dtype=dtype)

                    km = KMeans(n_clusters=nc, init='random',
                                n_init=1, max_iter=20, algorithm='full')

                    t = time.time()
                    km.fit(X)
                    t = time.time() - t
                    t /= km.n_iter_

                    ts[test, idx] = t
                    idx += 1

    tms = ts.mean(axis=0)
    tstds = ts.std(axis=0)

    idx = 0
    for ns in n_samples:
        for nc in n_clusters:
            if ns > nc:
                for nf in n_features:
                    print(str(dtype) + ',' + impl + ',' + str(ns) + ',' + str(nc) + ',' + str(nf) + ',' + str(tms[idx]) + ',' + str(tstds[idx]))
                    idx += 1


bench(np.float32, 0.01, 1, 'PR')


densities = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]


def bench_density(n_samples, n_clusters, n_features, n_tests, impl):
    n_tot = len(densities)
    ts = np.zeros((n_tests, n_tot))

    for test in range(n_tests):
        for idx, d in enumerate(densities):
            X = random(n_samples, n_features, density=d,
                       format='csr', dtype=np.float32)

            km = KMeans(n_clusters=n_clusters, init='random',
                        n_init=1, max_iter=20, algorithm='full')

            t = time.time()
            km.fit(X)
            t = time.time() - t
            t /= km.n_iter_

            ts[test, idx] = t

    tms = ts.mean(axis=0)
    tstds = ts.std(axis=0)

    for idx, d in enumerate(densities):
        print(impl + ',' + str(d) + ',' + str(tms[idx]) + ',' + str(tstds[idx]))


# bench_density(2**16, 2**6, 2**10, 5, 'intel')
# bench_density(2**10, 2**4, 2**2, 1, 'PR')
