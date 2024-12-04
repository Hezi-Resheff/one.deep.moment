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


if __name__ == "__main__":

    df = pd.DataFrame([])
    num_run = np.random.randint(1, 1000000)
    for ind in range(250):

        orig_size = 50   # This is the size of the PH the moments come from (so we know they are feasible)
        use_size = 50    # This is the size of the target PH
                  # This is the number of moments to match

        orig_size = np.random.randint(5, 100)
        use_size =  np.random.randint(max(5,orig_size-10), 100)
        n =  np.random.randint(5,21) #min(20,np.random.randint(5, 2*orig_size-1))

        print(orig_size, use_size, n)

        a, A, moms = sample(orig_size)
        fitted_moms = 5 #  np.random.randint(5, 21)
        moms = compute_first_n_moments(a, A, fitted_moms)
        ms = torch.tensor(np.array(moms).flatten())
        # ms = get_feasible_moments(original_size=orig_size, n=n)
        print(ms)
        ws = ms ** (-1)
        start = time.time()

        # matcher = MomentMatcher(ms)
        # (lambdas, ps, alpha), (a, T) = matcher.fit_ph_distribution(use_size, num_epochs=num_epochs, moment_weights=ws)

        # ms = get_feasible_moments(original_size=orig_size, n=n)
        print(ms)
        num_epochs = 4000000
        ws = ms ** (-1)

        matcher = MomentMatcher(ms)
        (lambdas, ps, alpha), (a, T) = matcher.fit_ph_distribution(use_size, num_epochs=num_epochs, moment_weights=ws)

        runtime = time.time() - start
        original_moments = ms.detach().numpy()
        computed_moments = [m.detach().item() for m in compute_moments(a, T, use_size, n)]
        moment_table = pd.DataFrame([computed_moments, original_moments], index="computed target".split()).T
        moment_table["delta"] = moment_table["computed"] - moment_table["target"]
        moment_table["delta-relative"] = 100*moment_table["delta"] / moment_table["target"]
        print(moment_table)
        if sys.platform == 'linux':

            path = '/scratch/eliransc/mom_match'
        else:
            path =  r'C:\Users\Eshel\workspace\data\mom_matching'

        file_name = 'num_run_' + str(num_run) + '.pkl'  #+ '_num_moms_' + str(n) + '_orig_size_' + str(orig_size) + '_use_size_' + str(
            #use_size) + '_epochs_' + str(num_epochs) + '_runtime_' + str(runtime) + '.pkl'
        full_path = os.path.join(path, file_name)

        curr_ind = df.shape[0]

        df.loc[curr_ind, 'runtime'] = runtime
        df.loc[curr_ind, 'numepochs'] = num_epochs
        df.loc[curr_ind, 'orig_PH_size'] = orig_size
        df.loc[curr_ind, 'fitted_PH_size'] = use_size
        df.loc[curr_ind, 'num_fitted_moms'] = n



        for ind in range(moment_table.shape[0]):
            df.loc[curr_ind, 'true_moms_' + str(ind + 1)] = moment_table.loc[ind, 'target']

        for ind in range(moment_table.shape[0]):
            df.loc[curr_ind, 'fitted_moms_' + str(ind + 1)] = moment_table.loc[ind, 'computed']

        for ind in range(moment_table.shape[0]):
            df.loc[curr_ind, 'error_' + str(ind + 1)] = moment_table.loc[ind, 'delta-relative']

        m1, m2, m3, m4 = df.loc[curr_ind, 'true_moms_1'], df.loc[curr_ind, 'true_moms_2'], df.loc[
            curr_ind, 'true_moms_3'], df.loc[
            curr_ind, 'true_moms_4']

        skew, kurt = compute_skewness_and_kurtosis_from_raw(m1, m2, m3, m4)

        df.loc[curr_ind, 'skewness'] = skew
        df.loc[curr_ind, 'kurtosis'] = kurt




        pkl.dump(df, open(full_path, 'wb'))
        print(df)


