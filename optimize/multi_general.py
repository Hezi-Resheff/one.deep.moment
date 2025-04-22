import os.path
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import time
import random


MIN_LOSS_EPSILON = 1e-9

class MomentMatcherBase(object):
    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lr_gamma=.9):
        self.k = ph_size
        self.n = n_replica
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.n_epochs = num_epochs
        self.params = None
        self.fit_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Starting with device: {self.device}")

    def fit(self, target_ms, stop=None):
        # init
        self._init()
        optimizer = torch.optim.Adam(self.params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)

        # train loop
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            loss, extended_loss_info = self._loss(target_ms.to(self.device))
            loss.backward()
            optimizer.step()

            losses = extended_loss_info["per_replica"]
            best_replica_loss = torch.min(losses[~torch.isnan(losses)])
            best_replica_ix = torch.argmin(losses[~torch.isnan(losses)])
            still_alive_count = torch.sum(torch.isfinite(losses))

            if epoch % 1000 == 999:
                scheduler.step()

            if epoch % 1000 == 999 or best_replica_loss < MIN_LOSS_EPSILON or epoch == self.n_epochs-1:
                print(f"===== Epoch: {epoch} =====")
                print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
                print(f"Overall loss: {loss}")
                print(f"Still alive: {still_alive_count}")
                print(f"Best replica (ix={best_replica_ix}): {best_replica_loss}")
                self.fit_info = extended_loss_info

            if best_replica_loss < MIN_LOSS_EPSILON:
                break

            if stop is not None:
                for level in stop:
                    if epoch == level["epoch"]:
                        # structure: {"when":epoch, "keep_fraction": num}
                        n_keep = int(level["keep_fraction"] * self.n)
                        ix_keep = torch.argsort(losses)[:n_keep]
                        self.params = tuple(el[ix_keep].clone().detach().to(self.device).requires_grad_(True) for el in self.params)
                        self.n = self.params[0].shape[0]
                        optimizer = torch.optim.Adam(self.params, lr=self.lr)
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)


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
        a = a.to(self.device)
        T = T.to(self.device)
        ms = self._compute_moments(a, T, n_moments=len(target_ms), device=self.device)
        weighted_error = (ms - target_ms.to(self.device)) / target_ms.to(self.device)
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

    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lr_gamma=.9,
                 normalize_m1=True,
                 init_drop=None):

        super().__init__(ph_size, n_replica, lr, num_epochs, lr_gamma)
        self.normalize_m1 = normalize_m1
        self.init_drop = init_drop

    def _init(self):
        device = self.device

        # sample P uniform on hte n-1 simplex (after the softmax transformation)
        ones = torch.ones(self.k)
        qs = torch.distributions.Dirichlet(ones).sample((self.n, self.k))
        ps = torch.log(qs).to(self.device)

        lambdas = torch.empty(self.n, self.k, requires_grad=False).to(self.device)
        lambdas.data = torch.rand(self.n, self.k).to(device) ** (.5) # self.ls

        alpha = torch.distributions.Dirichlet(ones).sample((self.n, 1)).squeeze(1).to(self.device)

        # calculate 1st moment and normalize lambdas
        if self.normalize_m1:
            self.params = alpha, lambdas, ps
            a, T = self. _make_phs_from_params()
            m1 = self._compute_moments(a, T, n_moments=1, device=device)
            lambdas.data = lambdas.data * m1**0.5

        # zero out some P values
        if self.init_drop is not None:
            if self.init_drop == "uniform":

                # first sample for each replica the p of drop, then sample the drop mask
                drop_prob = torch.rand(self.n)
                drop_mask = torch.empty(self.n, self.k, self.k)

                for i in range(self.n):
                    sub_mask = torch.bernoulli(torch.full((k,k), drop_prob[i]))
                    drop_mask[i] = sub_mask
            else:
                # all use the same proportion of drop
                drop_mask = torch.bernoulli(torch.full(ps.shape, self.init_drop))
                ps[drop_mask == 1] = 0.

        ps.requires_grad_(True)
        lambdas.requires_grad_(True)
        alpha.requires_grad_(True)

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


class HyperErlangMatcher(MomentMatcherBase):
    def __init__(self, block_sizes, n_replica=10, lr=1e-2, num_epochs=1000, lr_gamma=.9, sort_init=True,
                 normalize_m1=True):
        super().__init__(sum(block_sizes), n_replica, lr, num_epochs, lr_gamma)
        self.block_sizes = block_sizes
        self.n_blocks = len(block_sizes)
        self.sort_init = sort_init
        self.normalize_m1 = normalize_m1

    def _init(self):
        device = self.device

        lam = torch.rand(self.n, self.n_blocks).to(self.device)

        if self.sort_init:
            lam, _ = lam.sort(dim=1)

        lambdas = torch.empty(self.n, self.n_blocks, requires_grad=False).to(self.device)
        lambdas.data = lam

        ps = torch.randn(self.n, self.n_blocks, requires_grad=False).to(self.device)

        ones = torch.ones(self.n_blocks).to(self.device)
        alpha = torch.distributions.Dirichlet(ones).sample((self.n, 1)).squeeze(1).to(self.device)

        if self.normalize_m1:
            self.params = alpha, lambdas, ps
            a, T = self._make_phs_from_params()
            a = a.to(self.device)
            T = T.to(self.device)
            m1 = self._compute_moments(a, T, n_moments=1, device=device)
            lambdas.data = lambdas.data * m1 ** 0.5

        lambdas.requires_grad_(True)
        ps.requires_grad_(True)
        alpha.requires_grad_(True)
        self.params = alpha, lambdas, ps

    def _make_phs_from_params(self):
        alpha, lambdas, ps = self.params

        # Make a by softmax and adding in zeros
        a = torch.nn.functional.softmax(alpha, dim=1)
        aa = torch.zeros(self.n, self.k)
        cuma = torch.cumsum(torch.tensor(self.block_sizes), dim=0)

        for i in range(self.n_blocks):
            aa[:, cuma[i] - self.block_sizes[i]] = a[:, i]

        # lambda and ps are repeated according to block sizes
        ls = torch.cat([lambdas[:, [i]].repeat(1, bs) for i, bs in enumerate(self.block_sizes)], dim=1) ** 2
        ps = torch.cat([ps[:, [i]].repeat(1, bs) for i, bs in enumerate(self.block_sizes)], dim=1)
        ps = torch.sigmoid(ps)

        T = torch.diag_embed(-ls) + torch.diag_embed(ps[:, :-1] * ls[:, :-1], offset=1)

        return aa, T

class CoxianPHMatcher(MomentMatcherBase):
    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lr_gamma=.9, normalize_m1=True, sort_init=True):
        super().__init__(ph_size, n_replica, lr, num_epochs, lr_gamma)
        self.sort_init = sort_init
        self.normalize_m1 = normalize_m1

    def _init(self):
        device = self.device

        lam = torch.rand(self.n, self.k).to(self.device)

        if self.sort_init:
            lam, _ = lam.sort(dim=1)

        lambdas = torch.empty(self.n, self.k, requires_grad=False).to(self.device)
        lambdas.data = lam

        ps = torch.randn(self.n, self.k-1, requires_grad=False).to(self.device)

        if self.normalize_m1:
            self.params = lambdas, ps
            a, T = self._make_phs_from_params()
            m1 = self._compute_moments(a, T, n_moments=1, device=device)
            lambdas.data = lambdas.data * m1 ** 0.5

        lambdas.requires_grad_(True)
        ps.requires_grad_(True)
        self.params = lambdas, ps

    def _make_phs_from_params(self):
        lambdas, ps = self.params
        ls = lambdas ** 2

        p = torch.sigmoid(ps)

        a = torch.zeros(self.n, self.k).to(self.device)
        a[:, 0] = 1.

        T = torch.diag_embed(-ls) + torch.diag_embed(p * ls[:, :-1], offset=1)

        return a, T

def get_settings():

    try:
        df_run = pd.read_csv(r'C:\Users\Eshel\workspace\one.deep.moment\optimize_multi\run_lest.csv')
    except:
        df_run = pd.read_csv(os.path.abspath("../run_lest.csv" ))

    ind_selected = np.random.randint(df_run.shape[0])

    num_moms = int(df_run.loc[ind_selected, 'num_moms'].item())
    type_ph = df_run.loc[ind_selected, 'model']
    dataset = df_run.loc[ind_selected, 'dataset']
    k = int(df_run.loc[ind_selected, 'PH_size_fit'].item())

    init_drop_list = [0.9]

    init_drop = random.choice(init_drop_list)

    # num_moms = random.choice([5, 10, 20])

    # dataset = random.choice(['df_cox.csv',  'df_hyper.csv']) # 'df_general.csv',

    lr_gamma = random.choice([0.9])

    # type_ph = random.choice(['hyper', 'cox']) #  'cox', 'general'

    # if type_ph == 'general':
    #
    #     k = random.choice([80])
    #
    # elif type_ph == 'cox':
    #     k = random.choice([20, 50, 140])
    #
    # elif type_ph == 'hyper':
    #     k = random.choice([100])

    num_rep  = random.choice([10000])

    return (init_drop, num_moms, dataset, lr_gamma, type_ph, k, num_rep)


if __name__ == "__main__":
    try:
        from optimize.util import moment_analytics, compute_moments
    except:
        pass
    try:
        from util import moment_analytics, compute_moments

    except:
        pass


    ## Setup list:
    ### 1. Number of initial guesses.
    ### 2. Early stopping scheduler
    ### 3. Number of zeros in General PH generating matrix.
    ### 4. LR


    init_drop, num_moms, dataset, lr_gamma, type_ph, k, num_rep  = get_settings()
    print(init_drop, num_moms, dataset, lr_gamma, type_ph, k, num_rep)

    if (k > 50) & (num_rep == 10000):
        num_rep = 4000
        print('num_rep', num_rep)


    df_dat =  pd.read_csv(os.path.abspath("../optimize_multi/" +dataset))  #pkl.load(open(os.path.abspath("../optimize/" +dataset), 'rb'))
    print(df_dat.shape)

    df_res = pd.DataFrame([])
    # Initialize df_res if not already

    moms_cols = []
    for mom in range(1, num_moms + 1):
        moms_cols.append('mom_' + str(mom))

    rand_model = np.random.randint(1000000)

    num_epochs  = 80000
    for rand_ind in range(300):


        rand_ind = np.random.randint(df_dat.shape[0])

        try:

            if type_ph == 'general':

                m = GeneralPHMatcher(ph_size=k, num_epochs=num_epochs, lr=5e-3, n_replica=num_rep, lr_gamma=lr_gamma, normalize_m1=True,
                                    init_drop='uniform')
            elif type_ph == 'cox':

                m = CoxianPHMatcher(ph_size=k, num_epochs=num_epochs, lr=5e-3, lr_gamma=lr_gamma, n_replica=num_rep)

            elif type_ph == 'hyper':

                # k = np.array([10, 10,15,20,8,6,7,6]).sum()

                if  k == 20:
                    block_sizes = [4,6,3,7]
                elif  k == 50:
                    block_sizes = [4, 6, 3, 7, 10, 12, 8]
                else:
                    block_sizes = [4, 6, 3, 7, 10, 12, 8, 10, 20, 20,10]

                k = block_sizes.sum()



                m = HyperErlangMatcher(block_sizes=block_sizes, n_replica=num_rep, num_epochs=num_epochs, lr=5e-3, lr_gamma=lr_gamma)

            print(rand_ind)

            now = time.time()

            moments = torch.tensor(df_dat.loc[rand_ind, moms_cols])

            m.fit(target_ms=moments, stop=[{"epoch": 500, "keep_fraction": .2},
                                           {"epoch": 5000, "keep_fraction": .1},
                                           {"epoch": 15000, "keep_fraction": .1}
                                           ])

            a, T = m.get_best_after_fit()


            moment_table = moment_analytics(moments, compute_moments(a.to('cpu'), T.to('cpu'), k, len(moments)))
            print(moment_table)

            curr_ind = df_res.shape[0]

            end = time.time()
            runtime = end-now
            print('runtime is: ', runtime)

            df_res.loc[curr_ind,'run_time'] = runtime
            df_res.loc[curr_ind, 'k'] = k
            df_res.loc[curr_ind, 'type_ph'] = type_ph
            df_res.loc[curr_ind, 'type_test_ph'] = dataset
            df_res.loc[curr_ind, 'num_rep'] = num_rep
            df_res.loc[curr_ind, 'num_epochs'] = num_epochs
            df_res.loc[curr_ind, 'num_moms'] = num_moms
            df_res.loc[curr_ind, 'init_drop'] = init_drop
            df_res.loc[curr_ind, 'lr_gamma'] = lr_gamma
            df_res.loc[curr_ind, 'rand_ind'] = rand_ind

            for mom in range(1, num_moms + 1):
                df_res.loc[curr_ind, 'computed_' + str(mom)] = moment_table.loc[mom - 1, 'computed']

            for mom in range(1, num_moms + 1):
                df_res.loc[curr_ind, 'target_' + str(mom)] = moment_table.loc[mom - 1, 'target']

            for mom in range(1, num_moms + 1):
                df_res.loc[curr_ind, 'delta-relative_' + str(mom)] = moment_table.loc[mom - 1, 'delta-relative']

            file_name  = '222model_num_' + str(rand_model) +  '_df_res_type_ph_'+type_ph + '_init_drop_' + str(init_drop) +  '_ph_size_' + str(k) +  '_lr_gamma_' + str(lr_gamma)  +  '_nummoms_'   +str(num_moms)+'_testset_' + dataset[:-4] + '_size_'+str(k) + '_numrepli_'+str(num_rep) + '_num_epochs_'+str(num_epochs)+'.pkl'
            try:
                dump_path = '/scratch/eliransc'
                pkl.dump(df_res, open(os.path.join(dump_path,  file_name), 'wb'))
            except:
                pkl.dump(df_res, open(file_name, 'wb'))


        except:
            print('bad iteration', rand_ind)



