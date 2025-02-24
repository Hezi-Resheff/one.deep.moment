import os.path

import torch
import pickle as pkl
import numpy as np
import pandas as pd

MIN_LOSS_EPSILON = 1e-9


class MomentMatcherBase(object):
    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lambda_scale=10):
        self.k = ph_size
        self.n = n_replica
        self.lr = lr
        self.n_epochs = num_epochs
        self.ls = lambda_scale
        self.params = None
        self.fit_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Starting with device: {self.device}")

    def fit(self, target_ms):
        # init
        self._init()
        optimizer = torch.optim.Adam(self.params, lr=self.lr)

        # train loop
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            loss, extended_loss_info = self._loss(target_ms)
            loss.backward()
            optimizer.step()

            losses = extended_loss_info["per_replica"]
            best_replica_loss = torch.min(losses[~torch.isnan(losses)])
            still_alive_count = torch.sum(torch.isfinite(losses))
            if epoch % 100 == 99 or best_replica_loss < MIN_LOSS_EPSILON or epoch == self.n_epochs-1:
                print(f"===== Epoch: {epoch} =====")
                print(f"Overall loss: {loss}")
                print(f"Still alive: {still_alive_count}")
                print(f"Best replica: {best_replica_loss}")
                self.fit_info = extended_loss_info

            if best_replica_loss < MIN_LOSS_EPSILON:
                break

    @staticmethod
    def _compute_moments(a, T, n_moments, device):
        n, k, _ = T.shape
        m = []
        T_in = torch.inverse(T)
        T_powers = torch.eye(k).expand(n, k, k).to(device)
        signed_factorial = 1.
        one = torch.ones(k).to(device)

        for i in range(1, n_moments+1):
            signed_factorial *= -i
            T_powers = torch.matmul(T_powers, T_in)  # now T_powers is T^(-i)
            current_moment = signed_factorial * torch.einsum('bi,bij,j->b', a, T_powers, one)
            m.append(current_moment)

        return torch.stack(m).T

    def _loss(self, target_ms):
        a, T = self._make_phs_from_params()
        ms = self._compute_moments(a, T, n_moments=len(target_ms), device=self.device)
        weighted_error = (ms - target_ms) / target_ms
        per_replica_loss = torch.mean(weighted_error ** 2, dim=1)

        # Save all loss values
        extended_loss_info = {"per_replica": per_replica_loss.detach()}

        # Compute total loss without nan/inf (dead) replicas
        mask = ~torch.isnan(per_replica_loss) & ~torch.isinf(per_replica_loss)
        loss = per_replica_loss[mask].to(torch.float64).mean()

        return loss, extended_loss_info

    def get_best_after_fit(self):
        all_loss = self.fit_info["per_replica"]
        best_instance = torch.nan_to_num(all_loss, nan=float('inf')).argmin()
        a_all, T_all = self._make_phs_from_params()
        return a_all[best_instance], T_all[best_instance]


class GeneralPHMatcher(MomentMatcherBase):
    def _init(self):
        device = self.device
        ps = torch.randn(self.n, self.k, self.k, requires_grad=True).to(self.device)
        lambdas = torch.empty(self.n, self.k, requires_grad=True).to(self.device)
        lambdas.data = torch.rand(self.n, self.k).to(device) * self.ls
        alpha = torch.rand(self.n, self.k, requires_grad=True).to(self.device)
        self.params = alpha, lambdas, ps

    def _make_phs_from_params(self):
        alpha, lambdas, ps = self.params
        a = torch.nn.functional.softmax(alpha, dim=1)
        ls = lambdas ** 2
        lambda_rows = ls.unsqueeze(-1).expand(-1, -1, self.k)
        p = torch.nn.functional.softmax(ps, 2)

        diagonals = -torch.diagonal(p, dim1=1, dim2=2) - 1
        diagonals_back_to_3D = torch.diag_embed(diagonals)

        T = (p + diagonals_back_to_3D) * lambda_rows
        return a, T


class CoxianPHMatcher(MomentMatcherBase):
    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lambda_scale=10, sort_init=True):
        super().__init__(ph_size, n_replica, lr, num_epochs, lambda_scale)
        self.sort_init = sort_init

    def _init(self):
        lam = (torch.rand(self.n, self.k).to(self.device) * self.ls)
        if self.sort_init:
            lam, _ = lam.sort(dim=1)
        lambdas = torch.empty(self.n, self.k, requires_grad=True).to(self.device)
        lambdas.data = lam
        ps = torch.randn(self.n, self.k-1, requires_grad=True).to(self.device)
        self.params = lambdas, ps

    def _make_phs_from_params(self):
        lambdas, ps = self.params
        ls = lambdas ** 2

        p = torch.sigmoid(ps)

        a = torch.zeros(self.n, self.k).to(self.device)
        a[:, 0] = 1.

        T = torch.diag_embed(-ls) + torch.diag_embed(p * ls[:, :-1], offset=1)

        return a, T


if __name__ == "__main__":
    from optimize.util import moment_analytics, compute_moments
    df_res = pd.DataFrame([])
    # Initialize df_res if not already

    moms_cols = []
    num_moms = 10
    for mom in range(1, num_moms + 1):
        moms_cols.append('mom_' + str(mom))

    df_dat = pkl.load(open(os.path.abspath("../optimize/cox_df.pkl"), 'rb'))

    for rand_ind in range(df_dat.shape[0]):

        try:
            k = 10
            m = GeneralPHMatcher(ph_size=k, lambda_scale=10, num_epochs=15000, lr=5e-3, n_replica=5000)
            # m = CoxianPHMatcher(ph_size=k, lambda_scale=100, num_epochs=10000, lr=5e-3, n_replica=1000)

            print(rand_ind)

            moments = torch.tensor(df_dat.loc[rand_ind, moms_cols])
            m.fit(target_ms=moments)

            a, T = m.get_best_after_fit()
            # print(a)
            # print(T)

            moment_table = moment_analytics(moments, compute_moments(a, T, k, len(moments)))
            # print(moment_table)

            curr_ind = df_res.shape[0]
            for mom in range(1, num_moms + 1):
                df_res.loc[curr_ind, 'computed_' + str(mom)] = moment_table.loc[mom - 1, 'computed']

            for mom in range(1, num_moms + 1):
                df_res.loc[curr_ind, 'target_' + str(mom)] = moment_table.loc[mom - 1, 'target']

            for mom in range(1, num_moms + 1):
                df_res.loc[curr_ind, 'delta-relative_' + str(mom)] = moment_table.loc[mom - 1, 'delta-relative']

            pkl.dump(df_res, open('df_res_'+str(k) +'.pkl', 'wb'))

        except:
            print('bad iteration', rand_ind)



