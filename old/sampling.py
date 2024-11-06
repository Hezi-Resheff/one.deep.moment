
import numpy as np
import pandas as pd
from utils import compute_first_n_moments


def sample_coxian(degree, max_rate):
    lambdas = np.random.rand(degree) * max_rate
    ps = np.random.rand(degree - 1)
    A = np.diag(-lambdas) + np.diag(lambdas[:degree-1] * ps, k=1)
    alpha = np.eye(degree)[[0]]
    return alpha, A


def sample_flatten_log_moment(degree, max_rate, n, moment_index=2, clip=2.5, bins=250):
    # sample
    dists = []
    for _ in range(n):
        a, A = sample_coxian(degree=degree, max_rate=max_rate)
        A = A * compute_first_n_moments(a, A, n=2)[0]
        dists.append((a, A))

    # compute dist of moments and clip
    k_th_moments = [compute_first_n_moments(*d, n=2 * degree + 1)[moment_index] for d in dists]
    k_th_moments = np.array(k_th_moments)
    k_th_log_moments = np.log(k_th_moments)

    threshold_top = np.percentile(k_th_log_moments, 100-clip)
    threshold_bottom = np.percentile(k_th_log_moments, clip)

    clipped_dists = [d for d, moment in zip(dists, k_th_log_moments) if threshold_bottom <= moment <= threshold_top]
    clipped_k_th_log_moments = [moment for moment in k_th_log_moments if threshold_bottom <= moment <= threshold_top]

    # bin
    counts, bin_edges = np.histogram(clipped_k_th_log_moments, bins=20)
    bin_indices = np.digitize(clipped_k_th_log_moments, bin_edges[:-1])

    # sample
    sample_per_bin = pd.Series(bin_indices).value_counts().sort_index().min()
    use_dist_ix = pd.DataFrame(bin_indices, columns=["bin"]).groupby("bin").sample(sample_per_bin).index
    sampled_dist = [clipped_dists[i] for i in use_dist_ix]

    return sampled_dist









