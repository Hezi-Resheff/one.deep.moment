import torch
from util import *


# Stop optimization when the loss hits this value
MIN_LOSS_EPSILON = 1e-8


def make_ph(lambdas, ps, alpha, k):
    """ Use the arbitrary parameters, and make a valid PT representation  (a, T):
        lambdas: positive size k
        ps: size k x k
        alpha: size k
    """
    ls = lambdas ** 2
    a = torch.nn.functional.softmax(alpha, 0)
    p = torch.nn.functional.softmax(ps, 1)
    lambdas_on_rows = ls.repeat(k, 1).T
    T = (p + torch.diag(-1 - torch.diag(p))) * lambdas_on_rows
    return a, T


def compute_moments(a, T, k, n):
    """ generate first n moments of FT (a, T)
    m_i = ((-1) ** i) i! a T^(-i) 1
    """
    T_in = torch.inverse(T)
    T_powers = torch.eye(k)
    signed_factorial = 1.
    one = torch.ones(k)

    for i in range(1, n+1):
        signed_factorial *= -i
        T_powers = torch.matmul(T_powers, T_in)      # now T_powers is T^(-i)
        yield signed_factorial * a @ T_powers @ one


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


class MomentMatcher(object):
    def __init__(self, ms):
        self.ms = ms

    def fit_ph_distribution(self, k, num_epochs=1000, moment_weights=None,
                            lambda_scale=100, lr=1e-4, init=None):

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
            loss = compute_loss(ps, lambdas, alpha, k, self.ms, moment_weights)

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

        return (lambdas, ps, alpha), make_ph(lambdas, ps, alpha, k)


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

    ms = get_feasible_moments(original_size=100, n=20)
    ws = ms ** (-1)

    matcher = MomentMatcher(ms)
    out = matcher.fit_cascade(k_min=3, k_max=11, num_epochs=10000, moment_weights=ws, lambda_scale=10, lr=1e-4)

    (lambdas, ps, alpha), (a, T) = out[11]
    moment_table = moment_analytics(ms, compute_moments(a, T, 11, len(ms)))
    print(moment_table)

