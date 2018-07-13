import numpy as np
import sklearn
from sklearn.cluster import KMeans
import time

n_iter = 4

n_sample = 100000
n_feature = 50
n_component = 1000

#working_memory = [0.5, 1, 2, 4, 8, 12, 16, 20,
#                  24, 28, 32, 36, 40, 44, 48,
#                  52, 56, 60, 64, 68, 72, 76, 80,
#                  85, 90, 95, 100, 110, 120, 130, 140, 150,
#                  200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000]

working_memory = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  12, 14, 16, 18, 20, 25, 30, 40, 50,
                  60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]

points = np.random.random_sample((n_sample, n_feature)).astype(np.float32)
clusters = points[np.random.choice(n_sample, n_component, replace=False)]

with open('bench_kmeans_wm.csv', 'w') as file:
    file.write("wm,time\n")

    for exp in range(50):
        for i, wm in enumerate(working_memory):
            print((exp*len(working_memory) + i) / (50*len(working_memory)) * 100, "%")

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
