import sys
import square_dist as sqd
import numpy as np
import sklearn

n_iter = 20


n_sample_feature_component = [[1000,3,10],
                              [1000,3,100],
                              [1000,10,10],
                              [1000,10,100],
                              [1000,50,10],
                              [1000,50,100],
                              [10000,3,10],
                              [10000,3,100],
                              [10000,3,1000],
                              [10000,10,10],
                              [10000,10,100],
                              [10000,10,1000],
                              [10000,100,10],
                              [10000,100,100],
                              [10000,100,1000],
                              [10000,50,10],
                              [10000,50,100],
                              [10000,50,1000],
                              [100000,3,10],
                              [100000,3,100],
                              [100000,3,1000],
                              [100000,10,10],
                              [100000,10,100],
                              [100000,10,1000],
                              [100000,50,10],
                              [100000,50,100],
                              [100000,50,1000],
                              [100000,100,10],
                              [100000,100,100],
                              [100000,100,1000],
                              [1000000,3,10],
                              [1000000,3,100],
                              [1000000,3,1000],
                              [1000000,10,10],
                              [1000000,10,100],
                              [1000000,10,1000],
                              [1000000,50,10],
                              [1000000,50,100],
                              [1000000,50,1000],
                              [1000000,100,10],
                              [1000000,100,100],
                              [1000000,100,1000]]

sklearn.set_config(working_memory=7)

with open('bench_p_dist_funcs_avx2.csv','w') as file:
    file.write("n_sample,n_feature,n_component,func,gflops\n")

    for i, params in enumerate(n_sample_feature_component):
        print(i / len(n_sample_feature_component) * 100, "%")
        n_sample, n_feature, n_component = params

        points = np.random.random_sample((n_sample, n_feature)).astype('float32')

        ops, t = sqd.square_dist(sqd.pairwise_dist_sgemm_min_dist, points, n_component, n_iter)
        gflops = 1.0e-9 * ops / t
        s = str(n_sample) + "," + str(n_feature) + "," + str(n_component) + ",sgemm," + str(gflops) + "\n"
        file.write(s)

        ops, t = sqd.square_dist(sqd.pairwise_dist_matmul_min_dist, points, n_component, n_iter)
        gflops = 1.0e-9 * ops / t
        s = str(n_sample) + "," + str(n_feature) + "," + str(n_component) + ",matmul," + str(gflops) + "\n"
        file.write(s)

        ops, t = sqd.square_dist(sqd.sklearn_pairwise_dist_argmin, points, n_component, n_iter)
        gflops = 1.0e-9 * ops / t
        s = str(n_sample) + "," + str(n_feature) + "," + str(n_component) + ",sklearn," + str(gflops) + "\n"
        file.write(s)


"""
working_memory = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 250, 500, 750, 1000]
points = np.random.random_sample((100000, 50)).astype('float32')
with open('bench_p_dist_argmin_50.csv','w') as file:
    file.write("wm,gflops\n")

    for i, wm in enumerate(working_memory):
        print(i / len(working_memory) * 100, "%")

        sklearn.set_config(working_memory=wm)

        ops, t = sqd.square_dist(sqd.sklearn_pairwise_dist_argmin, points, 1024, n_iter)
        gflops = 1.0e-9 * ops / t

        s = str(wm) + "," + str(gflops) + "\n"
        file.write(s)

"""