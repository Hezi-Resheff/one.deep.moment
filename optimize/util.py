import torch
import pandas as pd

INF = 1e8


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


def embedd_next_ph(a, T, k):
    """ Embedd the order-k PH (a, T) in order (k+1)
    a' = [a; 0]
    T' = [[T 0], [0..., 1]]
    """
    a1 = torch.hstack([a, torch.Tensor([0.])])
    T1 = torch.zeros((k+1, k+1))
    T1[:-1, :-1] = T
    T1[-1, -1] = -1.
    return a1, T1


def embedd_next_parametrization(l, ps, a, k):
    a1 = torch.hstack([a, torch.Tensor([-INF])])
    l1 = torch.hstack([l, torch.Tensor([1.])])
    p1 = torch.ones(k+1, k+1) * -INF
    p1[:-1, :-1] = ps
    p1[-1, -1] = 1.
    return l1, p1, a1


def moment_analytics(ms, comp):
    original_moments = ms.detach().numpy()
    computed_moments = [m.detach().item() for m in comp]
    moment_table = pd.DataFrame([computed_moments, original_moments], index="computed target".split()).T
    moment_table["delta"] = moment_table["computed"] - moment_table["target"]
    moment_table["delta-relative"] = moment_table["delta"] / moment_table["target"]
    return moment_table


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


if __name__ == "__main__":
    from matching import *

    def example_embedd_ph():
        sep = "\n"+"-"*50+"\n"
        k = 3
        ps = torch.randn(k, k)
        lambdas = torch.rand(k)
        alpha = torch.randn(k)
        a, T = make_ph(lambdas, ps, alpha, k)
        print(a, sep, T, sep)
        print(list(compute_moments(a, T, k, 5)), sep)

        a1, T1 = embedd_next_ph(a, T, k)
        print(a1, sep, T1, sep)
        print(list(compute_moments(a1, T1, k+1, 5)), sep)

    def example_embedd_parametrization():
        sep = "\n" + "-" * 50 + "\n"
        k = 3
        ps = torch.randn(k, k)
        lambdas = torch.rand(k)
        alpha = torch.randn(k)
        l1, p1, a1 = embedd_next_parametrization(lambdas, ps, alpha, k)
        print(alpha, sep, a1, sep)
        print(lambdas, sep, l1, sep)
        print(ps, sep, p1, sep)

        a, T = make_ph(lambdas, ps, alpha, k)
        a1, T1 = make_ph(l1, p1, a1, k+1)
        print(a, sep, a1, sep)
        print(T, sep, T1, sep)

        print(sep, "The moments:", sep)
        print(list(compute_moments(a, T, k, 5)), sep)
        print(list(compute_moments(a1, T1, k+1, 5)), sep)

    # example_embedd_ph()
    example_embedd_parametrization()


