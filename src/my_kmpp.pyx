import time
import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX

def kmpp(X, n_clusters, x_squared_norms, n_local_trials):
    n_sample, n_feature = X.shape
    centers = np.empty((n_clusters, n_feature), dtype=np.float32)

    _kmpp(X, x_squared_norms, n_local_trials, centers)

# k-means++
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void _kmpp(np.ndarray[np.float32_t, ndim=2] X,
                 np.ndarray[np.float32_t, ndim=1] x_squared_norms,
                 int n_trials,
                 np.ndarray[np.float32_t, ndim=2] centers):

    cdef:
        int n_sample = X.shape[0]
        int n_feature = X.shape[1]
        int n_clusters = centers.shape[0]
        int n_local_trials = n_trials

    cdef:
        int first_center_id, best_candidate, candidate_id, r
        Py_ssize_t sample_idx, feature_idx, cluster_idx, trials_idx
        np.float32_t x, alpha, d1, d2
        np.float32_t current_pot, square_dist, best_pot, new_pot
        np.ndarray[np.float32_t, ndim=1] probas = np.empty(n_sample, dtype=np.float32)
        np.ndarray[np.int32_t, ndim=1] candidate_ids = np.zeros(n_local_trials, dtype=np.int32)
        np.ndarray[np.float32_t, ndim=1] closest_dist_sq = np.zeros(n_sample, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] new_dist_sq = np.zeros(n_sample, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] best_dist_sq = np.zeros(n_sample, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] distance_to_candidates = np.zeros((n_sample, n_local_trials), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] candidate = np.zeros(n_feature, dtype=np.float32)

    # Pick first center randomly
    first_center_id = np.random.randint(n_sample)
    centers[0] = X[first_center_id]

    with nogil:
        # Initialize list of closest distances and calculate current potential
        current_pot = 0.0
        for sample_idx in xrange(n_sample):
            square_dist = 0.0
            for feature_idx in xrange(n_feature):
                x = X[sample_idx, feature_idx] - centers[0, feature_idx]
                square_dist = square_dist + x * x
            closest_dist_sq[sample_idx] = square_dist
            current_pot += square_dist
        
        # loop
        for cluster_idx in xrange(1, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            alpha = 1 / current_pot
            for sample_idx in xrange(n_sample):
                probas[sample_idx] = closest_dist_sq[sample_idx] * alpha
            
            with gil:
                candidate_ids = np.random.choice(np.arange(n_sample), n_local_trials, p=probas, replace=False).astype(np.int32)
            
            # Compute distances to center candidates
            for sample_idx in prange(n_sample):
                for trials_idx in xrange(n_local_trials):
                    square_dist = 0.0
                    candidate_id = candidate_ids[trials_idx]
                    for feature_idx in xrange(n_feature):
                        x = X[sample_idx, feature_idx] - X[candidate_id, feature_idx]
                        square_dist = square_dist + x * x
                    distance_to_candidates[sample_idx, trials_idx] = square_dist
            
            # decide which candidate is the best
            best_candidate = -1
            best_pot = current_pot
            for sample_idx in xrange(n_sample):
                best_dist_sq[sample_idx] = closest_dist_sq[sample_idx]
            for trial_idx in xrange(n_local_trials):
                new_pot = 0.0
                for sample_idx in xrange(n_sample):
                    d1 = closest_dist_sq[sample_idx]
                    d2 = distance_to_candidates[sample_idx, trial_idx]
                    if d1 < d2:
                        new_dist_sq[sample_idx] = d1
                    else:
                        new_dist_sq[sample_idx] = d2
                    new_pot += new_dist_sq[sample_idx]
                
                if best_candidate == -1 or new_pot < best_pot:
                    best_candidate = candidate_ids[trial_idx]
                    best_pot = new_pot
                    for sample_idx in xrange(n_sample):
                        best_dist_sq[sample_idx] = new_dist_sq[sample_idx]
            
            # update clusters with best candidate
            for feature_idx in xrange(n_feature):
                centers[cluster_idx, feature_idx] = X[best_candidate, feature_idx]
            current_pot = best_pot
            for sample_idx in xrange(n_sample):
                closest_dist_sq[sample_idx] = best_dist_sq[sample_idx]
            