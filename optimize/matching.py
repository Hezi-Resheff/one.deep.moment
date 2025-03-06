import torch
# from pkg_resources import require
# from skopt import gp_minimize
# from skopt.space import Real, Integer
from util import *
import numpy as np
import pickle as pkl
import os
import sys
# Stop optimization when the loss hits this value
MIN_LOSS_EPSILON = 1e-9
# sys.path.append(os.path.abspath(".."))
# from utils_sample_ph import *


def compute_loss(ps, lambdas, alpha, k, ms, moment_weights=None):
    if moment_weights is None:
        moment_weights = torch.ones_like(ms)

    a, T = make_ph(lambdas, ps, alpha, k)
    moments = compute_moments(a, T, k, len(ms))
    moments = torch.stack(list(moments))

    error = (moments - ms)
    weighted_error = error * moment_weights
    ms_weighted_erorr = torch.mean(weighted_error ** 2)

    return ms_weighted_erorr


class CoxianMatcher(object):
    def __init__(self, ms):
        self.ms = ms

    @staticmethod
    def get_ph_from_coxiam(lams, ps, k):
        lams = lams ** 2
        ps = torch.sigmoid(ps)
        alpha = torch.eye(k)[0]
        T = torch.diag(-lams) + torch.diag(lams[:-1] * ps, 1)
        return alpha, T

    def compute_loss_cox(self, lams, ps, ws):
        k = lams.shape[0]
        a, T = self.get_ph_from_coxiam(lams, ps, k)
        moments = compute_moments(a, T, k, len(self.ms))
        moments = torch.stack(list(moments))

        error = (moments - self.ms)
        weighted_error = error * ws
        return torch.mean(weighted_error ** 2)

    def fit(self, k, num_epochs, moment_weights, lambda_scale, lr=1e-4):
        # init
        lam = (torch.randn(k) * lambda_scale).detach().requires_grad_(True)
        ps = torch.rand(k-1, requires_grad=True)

        # fit
        optimizer = torch.optim.Adam([lam, ps], lr=lr)
        loss_list = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss_cox(lam, ps, ws=moment_weights)
            loss_list.append(loss.item())

            if np.isnan(loss.item()):
                print('########## breaking - nan #########')

                break

            if loss < MIN_LOSS_EPSILON:
                break

            if len(loss_list) > 4000:
                if loss.item() > 10e4:
                    print('########## breaking - loss is too big #########')
                    break

            if len(loss_list) > 20000:
                if 100*np.abs((loss_list[-15000]-loss.item())/loss.item()) < 0.001:
                    print('########## breaking - stuck in local minumum #########')
                    break

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}: loss = {loss}")

        return loss.detach().item(), self.get_ph_from_coxiam(lam, ps, k)

    def fit_search_scale(self, k, num_epochs, moment_weights, lr=1e-4, max_scale = 50, min_scale = 1):

        loss = 1
        current_scale = max_scale
        best_so_far = (np.inf, (None, None))
        while current_scale > min_scale:
            loss_list = []
            print('##########################################')
            print('  Starting scale: ', current_scale)
            print('##########################################')

            current_loss, (a, T) = self.fit(k, num_epochs, moment_weights, current_scale)

            moments = compute_moments(a, T, k, len(self.ms))
            moments = torch.stack(list(moments)).detach().numpy().round(2)

            errors = 100 * torch.abs((torch.tensor(moments) - self.ms) / self.ms)
            print('###############################################################################################')
            print(errors.max().item())
            print('###############################################################################################')

            if current_loss < best_so_far[0]:
                best_so_far = (current_loss, (a, T))

            if current_loss < MIN_LOSS_EPSILON:
                return current_loss, (a, T)
            else:
                current_scale /= 2

        if np.isnan(best_so_far[0]):
            return 5000
        else:

            if errors.max().item() < 0.1:
                best_so_far = 1e-8

            return best_so_far

        # return best_so_far


class MultiErlangMomentMatcher(object):
    def __init__(self, ms, ls):
        self.ms = ms
        self.ls = ls
        self.k = sum(ls)

    def get_ph_mix_erlang(self, lam, alpha):
        lams = lam ** 2
        alphas = torch.nn.functional.softmax(alpha, 0)

        aT = [ErlangMomentMatcher.ph_from_lam(lmd, l)  for lmd, l in zip(lams, self.ls)]
        T = torch.block_diag(*[T[1] for T in aT])
        a = torch.cat([a[0] * a_ for a, a_ in zip(aT, alphas)])
        return a, T

    def compute_loss_mix_erlang(self, lam, alpha, ws):
        a, T = self.get_ph_mix_erlang(lam, alpha)
        moments = compute_moments(a, T, self.k, len(self.ms))
        moments = torch.stack(list(moments))

        error = (moments - self.ms)
        weighted_error = error * ws
        return torch.mean(weighted_error ** 2)

    def fit(self, num_epochs=1000, moment_weights=None, lambda_scale=100, lr=1e-4):
        # init
        lam = torch.tensor(torch.rand(len(self.ls)) * lambda_scale, requires_grad=True)
        alpha = torch.rand(len(self.ls), requires_grad=True)

        # fit
        optimizer = torch.optim.Adam([lam, alpha], lr=lr)
        loss_list  = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss_mix_erlang(lam, alpha, moment_weights)
            loss_list.append(loss.item())

            if loss < MIN_LOSS_EPSILON:
                break

            if np.isnan(loss.item()):
                print('########## breaking - nan #########')
                break

            if len(loss_list) > 10000:
                if loss.item() > 10e4:
                    print('########## breaking - loss is too big #########')
                    break

            if len(loss_list) > 20000:
                if 100*np.abs((loss_list[-15000]-loss.item())/loss.item()) < 0.01:
                    print('########## breaking - stuck in local minumum #########')
                    break

            elif loss < MIN_LOSS_EPSILON*10:
                lr = 1e-5

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}: loss = {loss}")
                if epoch % 10000 == 0:
                    a, T = self.get_ph_mix_erlang(lam, alpha)
                    moments = compute_moments(a, T, self.k, len(self.ms))
                    moments = torch.stack(list(moments)).detach().numpy().round(2)

                    print(f" => moments are: {moments}")
                    print(f" => true moments are: {self.ms}")
                    print(100 * (self.ms - moments) / self.ms)

        return loss.detach().item(), self.get_ph_mix_erlang(lam, alpha)

    def fit_search_scale(self, num_epochs=1000, moment_weights=None, lr=1e-4, max_scale=25, min_scale=5):
        loss = 1
        current_scale = max_scale

        best_so_far = (np.inf, (None, None))

        while current_scale > min_scale:
            loss_list = []
            print('##########################################')
            print('  Starting scale: ', current_scale)
            print('##########################################')
            current_loss, (a, T) = self.fit(num_epochs=num_epochs, moment_weights=moment_weights, lr=lr, lambda_scale=current_scale)

            moments = compute_moments(a, T, self.k, len(self.ms))
            moments = torch.stack(list(moments)).detach().numpy().round(2)

            errors = 100 * torch.abs((torch.tensor(moments) - self.ms) / self.ms)
            print('###############################################################################################')
            print(errors.max().item())
            print('###############################################################################################')

            if current_loss < best_so_far[0]:
                best_so_far = (current_loss, (a, T))

            if current_loss < MIN_LOSS_EPSILON:
                return current_loss, (a, T)
            else:
                current_scale /= 2

        if np.isnan(best_so_far[0]):
            return 5000
        else:

            if errors.max().item() < 0.1:
                best_so_far = 1e-8

            return best_so_far


class ErlangMomentMatcher(object):
    def __init__(self, ms):
        self.ms = ms

    def compute_loss_erlang(self, lam, k, moment_weights):
        a, T = self.ph_from_lam(lam, k)
        moments = compute_moments(a, T, k, len(self.ms))
        moments = torch.stack(list(moments))

        error = (moments - self.ms)
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

        return loss.detach().item(), self.ph_from_lam(lam, k)

    def fit_search_scale(self, num_epochs=1000, moment_weights=None, lr=1e-4, max_scale=100, min_scale=1):

        current_scale = max_scale

        best_so_far = (np.inf, (None, None))

        while current_scale > min_scale:
            loss_list = []
            print('##########################################')
            print('  Starting scale: ', current_scale)
            print('##########################################')
            current_loss, (a, T) = self.fit(num_epochs=num_epochs, moment_weights=moment_weights, lr=lr, lambda_scale=current_scale)

            if current_loss < best_so_far[0]:
                best_so_far = (current_loss, (a, T))

            if current_loss < MIN_LOSS_EPSILON:
                return current_loss, (a, T)
            else:
                current_scale /= 2


        if np.isnan(best_so_far[0]):
            return 5000
        else:
            return best_so_far


class MomentMatcher(object):
    def __init__(self, ms):
        self.ms = ms

    def fit_ph_distribution(self, k, num_epochs=1000, moment_weights=None,
                            lambda_scale=100, lr=1e-4, init=None):
        loss_history = []

        # init
        if init is None:
            ps = torch.randn(k, k, requires_grad=True)
            lambdas = torch.tensor(torch.rand(k)*lambda_scale, requires_grad=True)
            alpha = torch.rand(k, requires_grad=True)
        else:
            lambdas, ps, alpha = init
            lambdas = lambdas.detach().requires_grad_(True)
            ps = ps.detach().requires_grad_(True)
            alpha = alpha.detach().requires_grad_(True)

        # GD
        optimizer = torch.optim.Adam([alpha, lambdas, ps], lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = compute_loss(ps, lambdas, alpha, k, self.ms,  moment_weights )
            loss_history.append(loss.item())

            if np.isinf(loss.item()):
                print('########## breaking - loss is inf #########')
                break

            if np.isnan(loss.item()):
                print('########## breaking - nan #########')

                break


            if len(loss_history) > 10000:

                if loss.item() > 10e4:
                    print('########## breaking - loss is too big #########')
                    break

            elif len(loss_history) > 40000:
                if 100*np.abs((loss_history[-35000]-loss.item())/loss.item()) < 0.01:
                    print('########## breaking - stuck in local minumum #########')
                    break
                elif loss.item() > 10e1:
                    print('########## breaking - loss is too big #########')
                    break

            if np.isnan(loss.item()):
                print('########## breaking - nan #########')
                break

            if loss < MIN_LOSS_EPSILON:
                break

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0 or epoch == num_epochs-1:
                print(f"Epoch {epoch}: loss = {loss}")
                if epoch % 10000 == 0:
                    a, T = make_ph(lambdas, ps, alpha, k)
                    moments = compute_moments(a, T, k, len(self.ms))
                    moments = torch.stack(list(moments)).detach().numpy().round(2)

                    print(f" => moments are: {moments}")
                    print(f" => true moments are: {self.ms}")
                    print(100 * (self.ms - moments) / self.ms)

                    if (100 * (self.ms - moments) / self.ms).max().abs().item() < 0.2:
                        return loss.detach().item(),  make_ph(lambdas, ps, alpha, k)

                    if np.isnan(moments).sum() > 0:
                        return loss.detach().item(),  make_ph(lambdas, ps, alpha, k)

        return loss.detach().item(),  make_ph(lambdas, ps, alpha, k) # (lambdas, ps, alpha),

    def fit_search_scale(self, k,num_epochs=1000, moment_weights=None, lr=1e-4, max_scale=50, min_scale=1):
        loss = 1
        current_scale = max_scale

        best_so_far = (np.inf, (None, None))

        while current_scale > min_scale:
            loss_list = []
            print('##########################################')
            print('  Starting scale: ', current_scale)
            print('##########################################')

            current_loss, (a, T) = self.fit_ph_distribution(k, num_epochs=num_epochs, moment_weights=moment_weights, lr=lr,
                                            lambda_scale=current_scale)

            moments = compute_moments(a, T, k, len(self.ms))
            moments = torch.stack(list(moments)).detach().numpy().round(2)

            errors = 100*torch.abs((torch.tensor(moments)-self.ms)/self.ms)
            print('###############################################################################################')
            print(errors.max().item())
            print('###############################################################################################')

            if current_loss < best_so_far[0]:
                best_so_far = (current_loss, (a, T))

            if current_loss < MIN_LOSS_EPSILON:
                return current_loss, (a, T)
            else:
                current_scale /= 2

        if np.isnan(best_so_far[0]):
            return 5000
        else:
            if errors.max().item() < 0.1:
                best_so_far = 1e-8
            return best_so_far




# Temporary, so old code that uses this module doesn't break
def fit_ph_distribution(ms, **params):
    matcher = MomentMatcher(ms)
    return matcher.fit_ph_distribution(**params)


if __name__ == "__main__":
    from utils import compute_first_n_moments

    def stuff_eliran_was_running():

        ms = get_feasible_moments(original_size=100, n=10)
        ms = torch.tensor([1, 1.226187, 1.699542, 2.571434, 4.188616, 7.312320e+00, 1.367149e+01])
        ws = ms ** (-1)

        matcher = MomentMatcher(ms)
        out = matcher.fit_cascade(k_min=3, k_max=15, num_epochs=400000, moment_weights=ws, lambda_scale=10, lr=1e-4)

        k = 3
        (lambdas, ps, alpha), (a, T)= out[k]
        moment_table = moment_analytics(ms, compute_moments(a, T, k, len(ms)))

        path = r'C:\Users\Eshel\workspace\data\cascade'
        print(moment_table)
        num_run = np.random.randint(0, 10000)

        errors_mom = {}
        for key in out.keys():
            errors_mom[key] = (out[key][-2] - np.array(ms)) / np.array(ms)

        pkl.dump((errors_mom, np.array(ms)), open(os.path.join(path, str(num_run) + '_out.pkl'), 'wb'))


    # a, T, momenets = get_feasible_moments(original_size=20, n=5)
    moments = torch.tensor([1.00000000e+00, 1.76841773e+00, 4.03513940e+00, 1.14200072e+01,
       4.09989510e+01, 1.90084346e+02, 1.11890972e+03])

    print(moments)
    n_moments = len(moments)
    ls = np.array([10, 7, 10, 5, 4, 8, 12, 15,])
    k = 70 # ls.sum()
    num_epochs = 180000
    ws = moments ** (-1)

    # matcher = MultiErlangMomentMatcher(ms=moments, ls=ls)
    # loss, (a, T) = matcher.fit_search_scale(moment_weights=ws, num_epochs=num_epochs, lr=1e-4)

    matcher = CoxianMatcher(ms=moments)
    _, (a, T) = matcher.fit_search_scale(k, num_epochs=num_epochs, moment_weights=ws, lr=1e-4)

    moment_table = moment_analytics(moments, compute_moments(a, T, k, n_moments))
    print(moment_table)
    pkl.dump((a,T), open('a_T_ser_7.pkl', 'wb'))
