import numpy as np
import sklearn
from sklearn.cluster import KMeans
import time

n_iter = 20

n_sample = 100000
n_feature = 50
n_component = 1000

working_memory = [0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                  22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                  50, 60, 70, 80, 90, 100,
                  200, 300, 400, 500, 600, 700, 800, 900, 1000,
                  2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

points = np.random.random_sample((n_sample, n_feature)).astype(np.float32)
clusters = points[np.random.choice(n_sample, n_component, replace=False)]

with open('bench_kmeans_wm.csv', 'w') as file:
    file.write("wm,time\n")

    for i, wm in enumerate(working_memory):
        print(i / len(working_memory) * 100, "%")

        sklearn.set_config(working_memory=wm)

        t = time.time()
        km = KMeans(init=clusters,
                    n_init=1,
                    tol=1.0e-16,
                    n_clusters=n_component,
                    random_state=0,
                    max_iter=n_iter,
                    algorithm='full',
                    precompute_distances=True)
        km.fit(points)
        print(km.inertia_)
        t = time.time() - t

        s = str(wm) + "," + str(t) + "\n"
        file.write(s)
