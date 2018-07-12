import sys
import square_dist as sqd
import numpy as np
import sklearn

n_component = int(sys.argv[1])
n_iter = int(sys.argv[2])
func_id = int(sys.argv[3])

points = np.loadtxt("points.csv", delimiter=',', dtype=np.float32)

if func_id == 1:
    func = sqd.full_loop_min_dist_no_prange
elif func_id == 2:
    func = sqd.full_loop_min_dist_prange_sample
elif func_id == 3:
    working_memory = float(sys.argv[4])
    sklearn.set_config(working_memory=working_memory)
    func = sqd.sklearn_pairwise_dist_argmin
elif func_id == 4:
    func = sqd.pairwise_dist_full_loop_min_dist
elif func_id == 5:
    func = sqd.pairwise_dist_sgemm_min_dist
elif func_id == 6:
    func = sqd.pairwise_dist_matmul_min_dist

sqd.square_dist(func, points, n_component, n_iter)

