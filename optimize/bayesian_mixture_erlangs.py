import torch
from pkg_resources import require
from skopt import gp_minimize
from skopt.space import Real, Integer
from util import *
import numpy as np
import pickle as pkl
import os
import sys
# Stop optimization when the loss hits this value
MIN_LOSS_EPSILON = 1e-7
from matching import *

sys.path.append(os.path.abspath(".."))
from utils_sample_ph import *

from utils import *

def make_a_ph():
    """ sanity check for the make_ft function """
    k = 3
    ps = torch.randn(k, k)
    lambdas = torch.rand(k)
    alpha = torch.randn(k)
    a, T = make_ph(lambdas, ps, alpha, k)
    print("="*20)
    print("Sum of a: ", a.sum())
    print("="*20)
    print(T)
    print("Sum of T rows: ", T.sum(axis=1))
    print("=" * 20)

def compare_moment_methods():
    """ test the moment function... """
    k = 3
    ps = torch.randn(k, k)
    lambdas = torch.rand(k)
    alpha = torch.randn(k)
    a, T = make_ph(lambdas, ps, alpha, k)

    # External moment computation
    m_there = compute_first_n_moments(a, T, n=2*k-1)

    # This moment computation
    m_here = compute_moments(a, T, k, 2*k-1)

    # Compare
    for i, (m1, m2) in enumerate(zip(m_here, m_there)):
        print(f"Moment {i+1} is {m1:.3f} and {m2:.3f}")

# make_a_ph()
# compare_moment_methods()
def old_studd_no_longer_needed():
    ms = get_feasible_moments(original_size=100, n=10)
    ms = torch.tensor([1.        ,  1.88679887,  4.30712788, 10.73816182, 28.43716296])
    ws = ms ** (-1)

    matcher = MomentMatcher(ms)
    out = matcher.fit_cascade(k_min=3, k_max=15, num_epochs=400000, moment_weights=ws, lambda_scale=10, lr=1e-4)

    k = 3
    (lambdas, ps, alpha), (a, T)= out[k]
    moment_table = moment_analytics(ms, compute_moments(a, T, k, len(ms)))
    path = r'C:\Users\Eshel\workspace\data\cascade'
    print(moment_table)
    num_run = np.random.randint(0,10000)

    errors_mom = {}
    for key in out.keys():
        errors_mom[key]  = (out[key][-2] - np.array(ms)) / np.array(ms)

    pkl.dump((errors_mom, np.array(ms)), open(os.path.join(path, str(num_run) + '_out.pkl'), 'wb'))

# ms = torch.tensor([1.0000, 1.4652, 2.4290, 4.1820, 7.3312])
# ws = ms ** (-1)
# t = MultiErlangMomentMatcher(ms=ms, ls=[30, 70])
# loss, (a, T) = t.fit_search_scale(moment_weights=ws, num_epochs=60000, lr=5e-3)
# print(loss, a, T)

# ms = torch.tensor([1.0000,  1.3744,  2.3966,  4.8198, 10.5801])
ms1 = torch.tensor([1.        , 1.02777778, 1.08487654, 1.17528292, 1.30586991])
ms2 = torch.tensor([1., 1.88679887, 4.30712788, 10.73816182, 28.43716296])
ms3 = torch.tensor([1.0000,  1.2177,  2.1422,  6.4340, 30.2064])


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


class ErlangMomentMatcher(object):
    def __init__(self, ms):
        self.ms = ms

    def compute_loss_erlang(self, lam, k, moment_weights):
        a, T = self.ph_from_lam(lam, k)
        moments = compute_moments(a, T, k, len(self.ms))
        moments = torch.stack(list(moments))

        error = (moments - ms)
        weighted_error = error * moment_weights
        return torch.mean(weighted_error ** 2)

    @staticmethod
    def ph_from_lam(lam, k):
        l2 = lam ** 2  # parametrization trick, lam is now >= 0
        alpha = torch.eye(k)[0]
        T = torch.diag(-torch.ones(k) * l2) + torch.diag(torch.ones(k-1) * l2, 1)
        return alpha, T

    def fit(self, k, num_epochs=1000, moment_weights=None, lr=1e-4):
        # init
        lam = torch.randn(1, requires_grad=True)

        # fit
        optimizer = torch.optim.Adam([lam], lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss_erlang(lam, k, moment_weights)

            if loss < MIN_LOSS_EPSILON:
                break

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}: loss = {loss}")


# class MultiErlangMomentMatcher(object):
#     def __init__(self, ms, ls):
#         self.ms = ms
#         self.ls = ls
#         self.k = sum(ls)
#
#     def get_ph_mix_erlang(self, lam, alpha):
#         lams = lam ** 2
#         alphas = torch.nn.functional.softmax(alpha, 0)
#
#         aT = [ErlangMomentMatcher.ph_from_lam(lmd, l)  for lmd, l in zip(lams, self.ls)]
#         T = torch.block_diag(*[T[1] for T in aT])
#         a = torch.cat([a[0] * a_ for a, a_ in zip(aT, alphas)])
#         return a, T
#
#     def compute_loss_mix_erlang(self, lam, alpha, ws):
#         a, T = self.get_ph_mix_erlang(lam, alpha)
#         moments = compute_moments(a, T, self.k, len(self.ms))
#         moments = torch.stack(list(moments))
#
#         error = (moments - ms)
#         weighted_error = error * ws
#         return torch.mean(weighted_error ** 2)
#
#     def fit(self, num_epochs=1000, moment_weights=None, lambda_scale=100, lr=1e-4):
#         # init
#         lam = torch.tensor(torch.rand(len(self.ls)) * lambda_scale, requires_grad=True)
#         alpha = torch.rand(len(self.ls), requires_grad=True)
#
#         # fit
#         optimizer = torch.optim.Adam([lam, alpha], lr=lr)
#         loss_list  = []
#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             loss = self.compute_loss_mix_erlang(lam, alpha, moment_weights)
#             loss_list.append(loss.item())
#             if loss < MIN_LOSS_EPSILON:
#                 break
#
#             if np.isnan(loss.item()):
#                 print('########## breaking - nan #########')
#                 break
#
#             if len(loss_list) > 20000:
#                 if 100*np.abs((loss_list[-15000]-loss.item())/loss.item()) < 0.01:
#                     print('########## breaking - stuck in local minumum #########')
#                     break
#                 elif loss.item() > 10e4:
#                     print('########## breaking - loss is too big #########')
#                     break
#
#
#             elif loss < MIN_LOSS_EPSILON*10:
#                 lr = 1e-5
#
#             loss.backward()
#             optimizer.step()
#
#             if epoch % 1000 == 0 or epoch == num_epochs - 1:
#                 print(f"Epoch {epoch}: loss = {loss}")
#                 if epoch % 10000 == 0:
#                     a, T = self.get_ph_mix_erlang(lam, alpha)
#                     moments = compute_moments(a, T, self.k, len(self.ms))
#                     moments = torch.stack(list(moments)).detach().numpy().round(2)
#
#                     print(f" => moments are: {moments}")
#                     print(f" => true moments are: {self.ms}")
#                     print(100 * (self.ms - moments) / self.ms)
#
#         return loss.detach().item(), self.get_ph_mix_erlang(lam, alpha)
#
#     def fit_search_scale(self, num_epochs=1000, moment_weights=None, lr=1e-4, max_scale=50, min_scale=1):
#         loss = 1
#         current_scale = max_scale
#
#         best_so_far = (np.inf, (None, None))
#
#         while current_scale > min_scale:
#             loss_list = []
#             print('##########################################')
#             print('  Starting scale: ', current_scale)
#             print('##########################################')
#             current_loss, (a, T) = self.fit(num_epochs=num_epochs, moment_weights=moment_weights, lr=lr, lambda_scale=current_scale)
#
#             if current_loss < best_so_far[0]:
#                 best_so_far = (current_loss, (a, T))
#
#
#             if current_loss < MIN_LOSS_EPSILON:
#                 return current_loss, (a, T)
#             else:
#                 current_scale /= 2
#
#         return best_so_far

if sys.platform == 'linux':
    path_bayes_models = '/scratch/eliransc/bayes_models'
else:
    path_bayes_models = r'C:\Users\Eshel\workspace\data\bayes_models'




def cost_function(params):
    # Example: let's assume a simple quadratic cost function for demonstration
    ls = [l for l in params if l > 0]  # size 0 blocks don't count

    if np.array(params).sum() > max_val_ph:
        return 2.5

    print(f"    => Going with ls: {ls}")
    ws = ms ** (-1)
    t = MultiErlangMomentMatcher(ms=ms, ls=ls)
    loss, (a, T) = t.fit_search_scale(moment_weights=ws, num_epochs=62000, lr=5e-3)

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
    for mom in range(1,num_moms+1):
        df_res.loc[curr_ind, 'true_mom_'+str(mom)] = ms[mom-1].item()
        df_res.loc[curr_ind, 'est_mom_'+str(mom)] = moments[mom-1].item()
        df_res.loc[curr_ind, 'error_'+str(mom)] = errors[mom-1].item()

    # df_res.loc[curr_ind, 'true_mom_1'] = ms[0].item()
    # df_res.loc[curr_ind, 'true_mom_2'] = ms[1].item()
    # df_res.loc[curr_ind, 'true_mom_3'] = ms[2].item()
    # df_res.loc[curr_ind, 'true_mom_4'] = ms[3].item()
    # df_res.loc[curr_ind, 'true_mom_5'] = ms[4].item()
    #
    # df_res.loc[curr_ind, 'est_mom_1'] = moments[0].item()
    # df_res.loc[curr_ind, 'est_mom_2'] = moments[1].item()
    # df_res.loc[curr_ind, 'est_mom_3'] = moments[2].item()
    # df_res.loc[curr_ind, 'est_mom_4'] = moments[3].item()
    # df_res.loc[curr_ind, 'est_mom_5'] = moments[4].item()
    #
    #
    # df_res.loc[curr_ind, 'error_1'] = errors[0].item()
    # df_res.loc[curr_ind, 'error_2'] = errors[1].item()
    # df_res.loc[curr_ind, 'error_3'] = errors[2].item()
    # df_res.loc[curr_ind, 'error_4'] = errors[3].item()
    # df_res.loc[curr_ind, 'error_5'] = errors[4].item()

    print(df_res)

    dict_PH_per_iteration[curr_ind] = (a, T)

    pkl.dump(df_res, open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'wb'))
    pkl.dump(dict_PH_per_iteration,
             open(os.path.join(path_bayes_models, 'PH_dict_' + str(model_name) + '.pkl'), 'wb'))

    return loss



if sys.platform == 'linux':
    path_ph  = '/home/eliransc/projects/def-dkrass/eliransc/one.deep.moment'
    # df_dict_comb = pkl.load(open(os.path.join(path_ph, 'df_dict_comb.pkl'), 'rb'))
    # good_list_path = os.path.join(path_ph, 'good_list_general_experiment_new_mixture_test.pkl')


    df_dat = pkl.load(open(os.path.join(path_ph, 'general_df_lower_moms.pkl'), 'rb'))

else:
    path_ph = r'C:\Users\Eshel\workspace\data'
    path_ph = r'C:\Users\Eshel\workspace\data\mom_mathcher_data'
    # df_dat = pkl.load(open(os.path.join(path_ph, 'ph_size_20_moms.pkl'), 'rb'))
    # good_list_path = r'C:\Users\Eshel\workspace\one.deep.moment\old\good_list_ymca.pkl'
    # good_list_path = os.path.join(path_ph, 'good_list_20_moms_coxain_YMCA.pkl')
    df_dat = pkl.load(open(os.path.join(path_ph, 'general_df.pkl'), 'rb'))


model_type = 'mix_erlang'

for ind in range(1500):

    df_tot_res = pd.DataFrame([])

    run_num_tot = np.random.randint(1, 10000000)

    num_moms = np.random.choice([5,10, 20])
    max_val_ph = np.random.choice([20,50, 200])

    # list_keys = list(df_dict_comb.keys())

    # curr_key_ind = np.random.randint(len(list_keys))
    # df_dat = df_dict_comb[list_keys[curr_key_ind]]
    # good_list = pkl.load(open(good_list_path, 'rb'))

    rand_ind = np.random.randint(0,200) #np.random.choice(good_list[(max_val_ph, num_moms, list_keys[curr_key_ind])]).item()

    # good_list[(max_val_ph, num_moms, list_keys[curr_key_ind])] = good_list[(max_val_ph, num_moms, list_keys[curr_key_ind])][good_list[(max_val_ph, num_moms, list_keys[curr_key_ind])] != rand_ind]
    # pkl.dump(good_list, open(good_list_path, 'wb'))

    cols = []
    for mom in range(1, num_moms + 1):
        cols.append('mom_' + str(mom))

    ms = torch.tensor(df_dat.loc[rand_ind, cols].astype(float))

    time_start = time.time()

    model_name = np.random.randint(1,1000000)
    print('!!!!!!!!!!!!! new iteration !!!!!!!!!!!!!')

    print(model_name)

    df_res = pd.DataFrame([])
    pkl.dump(df_res, open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'wb'))

    dict_PH_per_iteration = {}

    pkl.dump(dict_PH_per_iteration, open(os.path.join(path_bayes_models, 'PH_dict_' + str(model_name) + '.pkl'), 'wb'))

    max_per_erlang = max_val_ph/2
    # Define the search space for each parameter
    space = [
        Integer(1, max_per_erlang, name='l1'),  # Continuous space for x1 between 0 and 10
        Integer(1, max_per_erlang, name='l2'),  # Integer space for x2 between 0 and 10
        Integer(1, max_per_erlang, name='l3'),  # Integer space for x3 between 0 and 10
        Integer(1, max_per_erlang, name='l4'),
        # Integer(1, 100, name='l5'), # Integer space for x4 between 0 and 10
    ]

    # Instantiate the stopping callback
    threshold = 1e-7
    stop_callback = StopWhenThresholdReached(threshold=threshold)


    # Perform Bayesian optimization with Gaussian Process
    result = gp_minimize(
        func=cost_function,  # The objective function to minimize
        dimensions=space,  # The search space
        n_calls=12,  # Number of evaluations of the objective function
        n_random_starts=5,
        callback=[print_score,stop_callback],  # Number of random starting points
        random_state=42  # Random seed for reproducibility
    )

    # Results
    print("Best cost found: ", result.fun)
    print("Best parameters found: ", result.x)

    res = pkl.load(open(os.path.join(path_bayes_models, str(model_name) + '.pkl'), 'rb'))

    error_cols = ['error_' + str(i) for i in range(1, num_moms + 1)]
    new_df = res[error_cols]

    new_df['mean_tot'] = new_df.mean(axis=1)

    ind_best = new_df['mean_tot'].argmin().item()

    curr_ind_tot = df_tot_res.shape[0]

    tot_time = time.time() - time_start

    for col in res.columns:
        df_tot_res.loc[curr_ind_tot, col] = res.loc[ind_best, col]

    df_tot_res.loc[curr_ind_tot, 'run_time'] = tot_time

    df_tot_res.loc[curr_ind_tot, 'ph_orig'] = df_dat.loc[rand_ind, 'orig_size'].astype(int)

    df_tot_res.loc[curr_ind_tot, 'orig_ind'] = rand_ind

    df_tot_res.loc[curr_ind_tot, 'num_moms'] = num_moms

    df_tot_res.loc[curr_ind_tot, 'PH_fit_size'] = max_val_ph

    # df_tot_res.loc[curr_ind_tot, 'key_ind'] = curr_key_ind


    if sys.platform == 'linux':

        path = '/scratch/eliransc/results_with_new_general'
        if not os.path.exists(path):
            os.mkdir(path)
        pkl.dump(df_tot_res, open(os.path.join(path, model_type+'_model_final_' + str(run_num_tot) + 'num_moms_'+str(num_moms) + '_max_PH_' + str(max_val_ph)  + '.pkl'), 'wb'))
    else:
        # path = r'C:\Users\Eshel\workspace\data\mom_matching_moms_10_cox'
        # pkl.dump(df_tot_res, open(os.path.join(path, 'model_final_' + str(run_num_tot) + 'num_moms_'+str(num_moms) + '_' + '.pkl'), 'wb'))

        path = r'C:\Users\Eshel\workspace\data\mom_mathcher_data\general_dataset_results'

        pkl.dump(df_tot_res,
                 open(os.path.join(path, model_type + '_model_final_' + str(run_num_tot) + 'num_moms_' + str(
                     num_moms) + '_max_PH_' + str(max_val_ph) + '.pkl'), 'wb'))


    # except:
    #     print('error in the bayesian optimiization')

