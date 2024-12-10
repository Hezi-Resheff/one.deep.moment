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
import pickle as pkl

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

def sample_PH(LB, UB, fitted_moms = 5):
    flag = True

    while flag:

        orig_size = np.random.randint(5, 101)
        a, A, moms = sample(orig_size)

        moms = compute_first_n_moments(a, A, fitted_moms)
        ms = torch.tensor(np.array(moms).flatten())
        if (ms[1]> LB) & (ms[1] < UB):
            flag = False

    return a, A, ms

if __name__ == "__main__":

    # if sys.platform == 'linux':
    #     bad_np, orig_size_arr = pkl.load( open('/home/eliransc/projects/def-dkrass/eliransc/one.deep.moment/bad_df.pkl', 'rb'))
    # else:
    #     bad_np, orig_size_arr = pkl.load( open(r'C:\Users\Eshel\workspace\data\bad_df.pkl', 'rb'))



    tot_df = pd.DataFrame([])
    num_run = np.random.randint(1, 1000000)
    for trial in range(250):
        df = pd.DataFrame([])
        dict_range = {0: [2, 6], 1: [6, 10], 2: [10, 14]}

        key = np.random.randint(len(dict_range.keys()))

        LB, UB = dict_range[key]
        fitted_moms =  5 # np.random.randint(5, 21)
        a, A, ms = sample_PH(LB, UB, fitted_moms)
        # ms = get_feasible_moments(original_size=orig_size, n=n)
        print(ms)
        # use_size = 20
        orig_size = A.shape[0]
        ws = ms ** (-1)
        # for index in range(250):

            # ind_example =  169 #np.random.randint(0,bad_np.shape[0])

        for use_size in  np.linspace(4,80, 20).astype(int):

            # orig_size = 5 # orig_size_arr[ind_example]
            # This is the size of the target PH
                      # This is the number of moments to match
            # ms = torch.tensor(bad_np[ind_example, ~np.isnan(bad_np[ind_example, :])])

            # ms = ms[:5]

            # n =  ms.shape[0] #np.random.randint(5,21) #min(20,np.random.randint(5, 2*orig_size-1))
            # orig_size = np.random.randint(5,101)
            # a, A, moms = sample(orig_size)
            # fitted_moms =   np.random.randint(5, 21)
            # moms = compute_first_n_moments(a, A, fitted_moms)
            # ms = torch.tensor(np.array(moms).flatten())


            start = time.time()

            # matcher = MomentMatcher(ms)
            # (lambdas, ps, alpha), (a, T) = matcher.fit_ph_distribution(use_size, num_epochs=num_epochs, moment_weights=ws)

            # ms = get_feasible_moments(original_size=orig_size, n=n)
            print(ms)
            num_epochs = 250000
            ws = ms ** (-1)

            matcher = MomentMatcher(ms)
            (lambdas, ps, alpha), (a, T) = matcher.fit_ph_distribution(use_size, num_epochs=num_epochs, moment_weights=ws)

            runtime = time.time() - start
            original_moments = ms.detach().numpy()
            computed_moments = [m.detach().item() for m in compute_moments(a, T, use_size, ms.shape[0])]
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
            df.loc[curr_ind, 'num_fitted_moms'] = ms.shape[0]

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

            if (moment_table["delta-relative"].abs().max()).item() < 0.2:
                break



        ind_tot = tot_df.shape[0]
        error_columns = [col for col in df.columns if col.startswith('error_')]
        best_ind  = df[error_columns].abs().mean(axis = 1).argmin().item()

        for col in df.columns:
            tot_df.loc[ind_tot, col] = df.loc[best_ind, col]
        tot_df.loc[ind_tot, 'tot_run_time'] = df.loc[:best_ind,'runtime'].sum()

        print('stop')

        if sys.platform == 'linux':
            pkl.dump(tot_df, open(os.path.join('/scratch/eliransc/mom_results_5_moments', str(num_run) +  '_tot_df.pkl'), 'wb'))
        else:
            pkl.dump(tot_df, open(os.path.join(r'C:\Users\Eshel\workspace\data\mom_match', str(num_run) + '_tot_df.pkl'), 'wb'))








