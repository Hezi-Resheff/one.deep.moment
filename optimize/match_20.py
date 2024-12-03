"""
Can we match 20 moments??
"""
import torch
import pandas as pd
from matching import *
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils_sample_ph import *


def compute_skewness_and_kurtosis_from_raw(m1, m2, m3, m4):
    # Compute central moments
    mu2 = m2 - m1**2
    mu3 = m3 - 3*m1*m2 + 2*m1**3
    mu4 = m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4

    # Compute skewness and kurtosis
    skewness = mu3 / (mu2 ** 1.5)
    kurtosis = mu4 / (mu2 ** 2)
    excess_kurtosis = kurtosis - 3

    return skewness, kurtosis

def get_feasible_moments(original_size, n):
    """ Compute feasible k-moments by sampling from high order PH and scaling """
    k = original_size
    ps = torch.randn(k, k)
    lambdas = torch.rand(k) * 100
    alpha = torch.randn(k)
    a, T = make_ph(lambdas, ps, alpha, k)

    # Compute mean
    ms = compute_moments(a, T, k, 1)
    m1 = torch.stack(list(ms)).item()

    # Scale
    T = T * m1
    ms = compute_moments(a, T, k, n)
    momenets = torch.stack(list(ms))
    return momenets
from util import get_feasible_moments


if __name__ == "__main__":
    orig_size = 50   # This is the size of the PH the moments come from (so we know they are feasible)
    use_size = 50    # This is the size of the target PH
              # This is the number of moments to match

    n = 20

    ms = get_feasible_moments(original_size=orig_size, n=n)
    print(ms)
    num_epochs = 400000
    ws = ms ** (-1)

    matcher = MomentMatcher(ms)
    (lambdas, ps, alpha), (a, T) = matcher.fit_ph_distribution(use_size, num_epochs=num_epochs, moment_weights=ws)

    moment_table = moment_analytics(ms, compute_moments(a, T, use_size, n))
    print(moment_table)


