import numpy as np
import sklearn
from sklearn.cluster import KMeans
import time

n_iter = 4

n_sample = 100000
n_feature = 50
n_component = 1000

working_memory = [0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                  22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                  45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                  200, 400, 600, 800, 1000,
                  2000, 4000, 6000, 8000, 10000]

points = np.random.random_sample((n_sample, n_feature)).astype(np.float32)
clusters = points[np.random.choice(n_sample, n_component, replace=False)]

with open('bench_kmeans_wm.csv', 'w') as file:
    file.write("wm,time\n")

    for exp in range(10):
        for i, wm in enumerate(working_memory):
            print((exp*len(working_memory) + i) / (10*len(working_memory)) * 100, "%")

            sklearn.set_config(working_memory=wm)

            t = time.time()
            km = KMeans(init=clusters,
                        n_init=1,
                        tol=0,
                        n_clusters=n_component,
                        random_state=0,
                        max_iter=n_iter,
                        algorithm='full',
                        precompute_distances=True)
            km.fit(points)
            t = time.time() - t

            s = str(wm) + "," + str(t) + "\n"
            file.write(s)
