"""
Can we match 20 moments??
"""
import torch
import pandas as pd
from matching import *


def get_feasible_moments(original_size, n):
    """ Compute feasible k-moments by sampling from high order PH and scaling """
    k = original_size
    ps = torch.randn(k, k)
    lambdas = torch.rand(k)
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


if __name__ == "__main__":
    orig_size = 100
    use_size = 50
    n = 20

    ms = get_feasible_moments(original_size=orig_size, n=n)
    print(ms)

    ws = ms ** (-1)

    (lambdas, ps, alpha), (a, T) = fit_ph_distribution(ms, use_size, num_epochs=200000, moment_weights=ws)

    original_moments = ms.detach().numpy()
    computed_moments = [m.detach().item() for m in compute_moments(a, T, use_size, n)]
    moment_table = pd.DataFrame([computed_moments, original_moments], index="computed target".split()).T
    moment_table["delta"] = moment_table["computed"] - moment_table["target"]
    moment_table["delta-relative"] = moment_table["delta"] / moment_table["target"]
    print(moment_table)
