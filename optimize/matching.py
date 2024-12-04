import torch
from util import *
import numpy as np

# Stop optimization when the loss hits this value
MIN_LOSS_EPSILON = 1e-8


def compute_loss(ps, lambdas, alpha, k, ms,epoch, moment_weights=None):
    if moment_weights is None:
        moment_weights = torch.ones_like(ms)

    a, T = make_ph(lambdas, ps, alpha, k)
    moments = compute_moments(a, T, k, len(ms))
    moments = torch.stack(list(moments))

    error = (moments - ms)
    if epoch < 250000:
        weighted_error = error * moment_weights
    elif epoch < 300000:
        weighted_error = error * moment_weights**0.5
    else:
        weighted_error = error*1

    ms_weighted_erorr = torch.mean(weighted_error ** 2)

    return ms_weighted_erorr


class MomentMatcher(object):
    def __init__(self, ms):
        self.ms = ms

    def fit_ph_distribution(self, k, num_epochs=1000, moment_weights=None,
                            lambda_scale=100, lr=1e-4, init=None):

        # lambda_scale = 1000
        # init
        if init is None:
            ps = torch.randn(k, k, requires_grad=True)
            lambdas = torch.tensor(torch.rand(k)*lambda_scale, requires_grad=True)
            alpha = torch.randn(k, requires_grad=True)
        else:
            lambdas, ps, alpha = init
            lambdas = lambdas.detach().requires_grad_(True)
            ps = ps.detach().requires_grad_(True)
            alpha = alpha.detach().requires_grad_(True)

        # GD
        optimizer = torch.optim.Adam([alpha, lambdas, ps], lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = compute_loss(ps, lambdas, alpha, k, self.ms, epoch , moment_weights )

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

                    if np.isnan(moments).sum() > 0:
                        return (lambdas, ps, alpha), make_ph(lambdas, ps, alpha, k)


        return (lambdas, ps, alpha), make_ph(lambdas, ps, alpha, k)

    def fit_cascade(self, k_min, k_max, num_epochs=1000, moment_weights=None,
                            lambda_scale=100, lr=1e-4, init=None):

        out = {}

        ps = torch.randn(k_min, k_min)
        lambdas = torch.rand(k_min) * lambda_scale
        alpha = torch.randn(k_min, requires_grad=True)

        for k in range(k_min, k_max+1):
            print(f"==> Trying with k = {k}...")
            init = lambdas, ps, alpha
            this_out = self.fit_ph_distribution(k, num_epochs=num_epochs, moment_weights=moment_weights,
                                                lambda_scale=lambda_scale, lr=lr, init=init)

            (lambdas, ps, alpha), (a, T) = this_out
            out[k] = this_out
            lambdas, ps, alpha = embedd_next_parametrization(lambdas, ps, alpha, k)

        return out


# Temporary, so old code that uses this module doesn't break
def fit_ph_distribution(ms, **params):
    matcher = MomentMatcher(ms)
    return matcher.fit_ph_distribution(**params)


if __name__ == "__main__":
    from utils import compute_first_n_moments

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

    ms = get_feasible_moments(original_size=100, n=10)
    ws = ms ** (-1)

    matcher = MomentMatcher(ms)
    out = matcher.fit_cascade(k_min=3, k_max=3, num_epochs=100000, moment_weights=ws, lambda_scale=10, lr=1e-4)

    k = 3
    (lambdas, ps, alpha), (a, T) = out[k]
    moment_table = moment_analytics(ms, compute_moments(a, T, k, len(ms)))
    print(moment_table)

