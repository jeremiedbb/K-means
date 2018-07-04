import sys
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import time

n_iter = 10

working_memory = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 250, 500, 1000, 2500, 5000, 10000]
points = np.random.random_sample((100000, 50)).astype('float32')
with open('bench_kmeans_wm.csv','w') as file:
    file.write("wm,time\n")

    for i, wm in enumerate(working_memory):
        print(i / len(working_memory) * 100, "%")

        sklearn.set_config(working_memory=wm)
    
        t = time.time()
        km = KMeans(init='random',
                    n_init=1,
                    tol=1.0e-16,
                    n_clusters=1000,
                    random_state=0,
                    max_iter=n_iter,
                    algorithm='full',
                    precompute_distances=True)
        km.fit(points)
        print(km.inertia_)
        t = time.time() - t

        s = str(wm) + "," + str(t) + "\n"
        file.write(s)
