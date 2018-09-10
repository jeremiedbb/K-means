import numpy as np
from kmeans_chunked import kmeans_lloyd_chunked
from sklearn.cluster import KMeans
import time


def kmeans(X, centers, n_samples_chunk, max_iter):
    x_squared_norms = (X**2).sum(axis=1)

    centers_old = np.zeros_like(centers)
    centers_new = centers.copy()
    labels = np.zeros(X.shape[0], dtype=np.int32)

    for i in range(max_iter):

        inertia = kmeans_lloyd_chunked(X,
                                       x_squared_norms,
                                       centers_old,
                                       centers_new,
                                       labels,
                                       n_samples_chunk)

        center_shift = ((centers_new - centers_old)**2).sum()
        if center_shift <= 0.0:
            break

    if center_shift > 0:
        inertia = kmeans_lloyd_chunked(X,
                                       x_squared_norms,
                                       centers_old,
                                       centers_new,
                                       labels,
                                       n_samples_chunk)

    return labels, inertia, centers_old, i + 1


def bench_one(n_samples, n_clusters, n_features, L3, n_tests):
    X = np.random.random_sample((n_samples, n_features)).astype(np.float32)
    centers = X[np.random.choice(np.arange(X.shape[0]), n_clusters)]

    n_samples_chunk = int(L3 * 2**18 / (2 * n_clusters))

    ts = np.zeros(n_tests)
    for test in range(n_tests):
        t = time.time()
        labels, inertia, centers_old, n_iter = \
            kmeans(X, centers, n_samples_chunk, 20)
        t = time.time() - t
        t /= n_iter
        ts[test] = t

    tm = ts.mean()
    tstd = ts.std()
    print(str(tm) + ',' + str(tstd))


def bench_chunk_size(n_samples, n_clusters, n_features, n_tests):
    X = np.random.random_sample((n_samples, n_features)).astype(np.float32)
    centers = X[np.random.choice(np.arange(X.shape[0]), n_clusters)]

    chunk_sizes = [2**(j) for j in np.arange(3, 1+int(np.log2(n_samples)))]
    ts = np.zeros((n_tests, len(chunk_sizes)))

    for test in range(n_tests):
        for i, n_samples_chunk in enumerate(chunk_sizes):
            t = time.time()
            labels, inertia, centers_old, n_iter = \
                kmeans(X, centers, n_samples_chunk, 20)
            t = time.time() - t
            t /= n_iter
            ts[test, i] = t

    tms = ts.mean(axis=0)
    tstds = ts.std(axis=0)

    for i, n_samples_chunk in enumerate(chunk_sizes):
        print(str(n_samples_chunk) + ',' + str(tms[i]) + ',' + str(tstds[i]))


def bench_data_size(n_tests, L3, impl):
    n_samples = [2**i for i in np.arange(10, 20)]
    n_clusters = [2**j for j in np.arange(3, 10)]
    n_features = [2**k for k in np.arange(1, 7)]

    n_tot = 0
    for ns in n_samples:
        for nc in n_clusters:
            if ns > nc:
                for nf in n_features:
                    n_tot += 1

    ts = np.zeros((n_tests, n_tot))
    for test in range(n_tests):
        idx = 0
        for ns in n_samples:
            for nc in n_clusters:
                if ns > nc:
                    for nf in n_features:
                        X = np.random.random_sample((ns, nf)).astype(np.float32)
                        centers = X[np.random.choice(np.arange(X.shape[0]), nc)]

                        n_samples_chunk = 2**9

                        t = time.time()

                        if impl == 'PR':
                            labels, inertia, centers_old, n_iter = \
                                kmeans(X, centers, n_samples_chunk, 20)
                        elif impl == 'master':
                            km = KMeans(n_clusters=nc, init=centers, n_init=1, max_iter=20,
                                        precompute_distances=True, tol=0, algorithm='full')
                            km.fit(X)
                            n_iter = km.n_iter_
                        elif impl == 'intel':
                            km = KMeans(n_clusters=int(nc), init=centers, n_init=1, max_iter=20,
                                        tol=0, algorithm='full')
                            km.fit(X)
                            n_iter = km.n_iter_

                        t = time.time() - t
                        t /= n_iter
                        ts[test, idx] = t
                        idx += 1

    tms = ts.mean(axis=0)
    tstds = ts.std(axis=0)

    idx = 0
    for ns in n_samples:
        for nc in n_clusters:
            if ns > nc:
                for nf in n_features:
                    print(impl + ',' + str(ns) + ',' + str(nc) + ',' + str(nf) + ',' + str(tms[idx]) + ',' + str(tstds[idx]))
                    idx += 1


#bench_one(2**17, 2**9, 2**1, 4, 20)
bench_chunk_size(2**18, 2**12, 2**1, 3)
#bench_data_size(20, 4, 'intel')