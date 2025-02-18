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
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# num_moms = 10

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


if sys.platform == 'linux':
    path_bayes_models = '/scratch/eliransc/bayes_models_classic'
else:
    path_bayes_models = '.' #r'C:\Users\Eshel\workspace\data\bayes_models_classic'



def cost_function(params):


    print(f"    => Going with ls: {params}")


    num_epochs = 150000
    ws = ms ** (-1)
    matcher = CoxianMatcher(ms=ms)

    loss, (a, T) = matcher.fit_search_scale(params[0], num_epochs=num_epochs, moment_weights=ws, lr=1e-4)


    try:
        moments = compute_moments(a, T, T.shape[0], len(ms))
        moments = torch.stack(list(moments)).detach().numpy().round(2)

        print(f" => moments are: {moments}")
        print(f" => true moments are: {ms}")

        errors = 100 * (ms - moments) / ms

        print(errors)
        df_res = pkl.load(open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'rb'))

        dict_PH_per_iteration = pkl.load(
            open(os.path.join(path_bayes_models, 'PH_dict_' + str(model_name) + '.pkl'), 'rb'))

        curr_ind = df_res.shape[0]

        errors = errors.abs()
        for mom in range(1, num_moms + 1):
            df_res.loc[curr_ind, 'true_mom_' + str(mom)] = ms[mom - 1].item()
            df_res.loc[curr_ind, 'est_mom_' + str(mom)] = moments[mom - 1].item()
            df_res.loc[curr_ind, 'error_' + str(mom)] = errors[mom - 1].item()


        print(df_res)

        dict_PH_per_iteration[curr_ind] = (a, T)

        pkl.dump(df_res, open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'wb'))
        pkl.dump(dict_PH_per_iteration,
                 open(os.path.join(path_bayes_models, 'PH_dict_' + str(model_name) + '.pkl'), 'wb'))

        return loss

    except:

        return 500

def print_score(res):
    iteration = len(res.func_vals)
    current_score = res.func_vals[-1]
    sol = res.x
    print(f"Iteration {iteration}: Current score = {current_score} solution =  {sol}")



class StopWhenThresholdReached:
    def __init__(self, threshold):
        self.threshold = threshold  # The desired stopping criterion (value)

    def __call__(self, res):
        # Check the best function value so far
        best_value = res.fun
        if best_value <= self.threshold:
            print(f"Threshold reached: {best_value} <= {self.threshold}")
            return True  # Stop optimization
        return False  # Continue optimization

if __name__ == "__main__":



    model_type = 'cox'
    if sys.platform == 'linux':
        path_ph = '/home/eliransc/projects/def-dkrass/eliransc/one.deep.moment'
        df_dict_comb = pkl.load(open(os.path.join(path_ph, 'df_dict_comb.pkl'), 'rb'))
        good_list_path = os.path.join(path_ph, 'good_list_general_experiment_new_cox_test.pkl')

    else:
        path_ph = '.' #r'C:\Users\Eshel\workspace\data\mom_mathcher_data'
        # df_dat = pkl.load(open(os.path.join(path_ph, 'ph_size_20_moms.pkl'), 'rb'))
        # df_dat = pkl.load(open(os.path.join(path_ph, 'general_df.pkl'), 'rb'))


    for ind in range(1500):

        df_tot_res = pd.DataFrame([])

        run_num_tot = np.random.randint(1, 10000000)

        num_moms = np.random.choice([5])
        max_val_ph = np.random.choice([20,50, 200])

        dataset_file = np.random.choice(['cox_df.pkl']) # , 'general_df.pkl'

        df_dat = pkl.load(open(os.path.join(path_ph, dataset_file), 'rb'))

        # list_keys = list(df_dict_comb.keys())

        # curr_key_ind = np.random.randint(len(list_keys))
        # df_dat = df_dict_comb[list_keys[curr_key_ind]]
        # good_list = pkl.load(open(good_list_path, 'rb'))

        rand_ind = np.random.randint(0,200) # np.random.choice(good_list[(max_val_ph, num_moms, list_keys[curr_key_ind])]).item()


        cols = []
        for mom in range(1, num_moms + 1):
            cols.append('mom_' + str(mom))

        ms = torch.tensor(df_dat.loc[rand_ind, cols].astype(float))

        time_start = time.time()

        model_name = np.random.randint(1, 1000000)
        print('!!!!!!!!!!!!! new iteration !!!!!!!!!!!!!')

        print(model_name)

        df_res = pd.DataFrame([])
        pkl.dump(df_res, open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'wb'))

        dict_PH_per_iteration = {}

        pkl.dump(dict_PH_per_iteration,
                 open(os.path.join(path_bayes_models, 'PH_dict_' + str(model_name) + '.pkl'), 'wb'))

        # Define the search space for each parameter
        space = [
            Integer(5, max_val_ph, name='use_size'),  # Continuous space for x1 between 0 and 10
        ]

        # Instantiate the stopping callback
        threshold = 1e-7
        stop_callback = StopWhenThresholdReached(threshold=threshold)

        # Perform Bayesian optimization with Gaussian Process
        result = gp_minimize(
            func=cost_function,  # The objective function to minimize
            dimensions=space,  # The search space
            n_calls=10,  # Number of evaluations of the objective function
            n_random_starts=3,
            callback=[print_score, stop_callback],  # Number of random starting points
            random_state=42  # Random seed for reproducibility
        )

        # Results
        print("Best cost found: ", result.fun)
        print("Best parameters found: ", result.x)

        res = pkl.load(open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'rb'))

        error_cols = ['error_' + str(i) for i in range(1, num_moms+1)]
        new_df = res[error_cols]

        new_df['mean_tot'] = new_df.mean(axis=1)

        ind_best = new_df['mean_tot'].argmin().item()

        curr_ind_tot = df_tot_res.shape[0]

        tot_time = time.time() - time_start

        for col in res.columns:
            df_tot_res.loc[curr_ind_tot, col] = res.loc[ind_best, col]

        df_tot_res.loc[curr_ind_tot, 'run_time'] = tot_time

        df_tot_res.loc[curr_ind_tot, 'ph_orig'] = df_dat.loc[rand_ind, 'orig_size'].astype('int')

        df_tot_res.loc[curr_ind_tot, 'orig_ind'] = rand_ind

        df_tot_res.loc[curr_ind_tot, 'num_moms'] = num_moms

        df_tot_res.loc[curr_ind_tot, 'PH_fit_size'] = max_val_ph



        if sys.platform == 'linux':

            path = '/scratch/eliransc/experiment_type_num_moms_max_ph_with_cox_a'

            if not os.path.exists(path):
                os.mkdir(path)

            pkl.dump(df_tot_res, open(os.path.join(path, model_type + '_model_final_' + str(run_num_tot) + 'num_moms_'+str(num_moms) + '_max_PH_' + str(max_val_ph) + '.pkl'), 'wb'))
        else:

            path_dump = r'C:\Users\Eshel\workspace\data\mom_mathcher_data\all_datasets_all_models'

            pkl.dump(df_tot_res,
                     open(os.path.join(path_dump, 'dataset_' + dataset_file[:-4] + '_' + model_type + '_model_final_' + str(
                         run_num_tot) + 'num_moms_' + str(
                         num_moms) + '_max_PH_' + str(max_val_ph) + '.pkl'), 'wb'))
            # path = r'C:\Users\Eshel\workspace\data\mom_matching_bayes_classic_cox_'+str(num_moms)
            # pkl.dump(df_tot_res, open(os.path.join(path, 'model_final_' + str(run_num_tot) + '.pkl'), 'wb'))






