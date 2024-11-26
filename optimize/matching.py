import torch


def make_ft(lambdas, ps, alpha, k):
    """ Use the arbitrary parameters, and make a valid FT representation  (a, T):
        lambdas: positive size k
        ps: size k x k
        alpha: size k
    """
    a = torch.nn.functional.softmax(alpha, 0)
    p = torch.nn.functional.softmax(ps, 1)
    lambdas_on_rows = lambdas.repeat(k, 1).T
    T = (p + torch.diag(-1 - torch.diag(p))) * lambdas_on_rows
    return a, T


def compute_moments(a, T, k, n):
    """ generate first n moments of FT (a, T)
    m_i = ((-1) ** i) i! a T^(-i) 1
    """
    T_in = torch.inverse(T)
    T_powers = torch.eye(k)
    signed_factorial = 1
    one = torch.ones(k)

    for i in range(1, n+1):
        signed_factorial *= -i
        T_powers = torch.matmul(T_powers, T_in)      # now T_powers is T^(-i)
        yield signed_factorial * a @ T_powers @ one


def compute_loss(ps, lambdas, alpha, k, ms):
    a, T = make_ft(lambdas, ps, alpha, k)
    moments = compute_moments(a, T, k, len(ms))
    moments = torch.stack(list(moments))
    return torch.mean((moments - ms) ** 2)


def fit_ft_distribution(ms, k, num_epochs=1000):

    # init
    ps = torch.randn(k, k, requires_grad=True)
    lambdas = torch.rand(k, requires_grad=True)  # these must stay positive
    alpha = torch.randn(k, requires_grad=True)

    # GD
    optimizer = torch.optim.Adam([alpha, lambdas, ps], lr=0.0001)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = compute_loss(ps, lambdas, alpha, k, ms)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: loss = {loss}")
            if epoch % 10000 == 0:
                a, T = make_ft(lambdas, ps, alpha, k)
                moments = compute_moments(a, T, k, len(ms))
                moments = torch.stack(list(moments)).detach().numpy().round(2)
                print(f" => moments are: {moments}")


if __name__ == "__main__":
    from utils import compute_first_n_moments

    def make_a_ft():
        """ sanity check for the make_ft function """
        k = 3
        ps = torch.randn(k, k)
        lambdas = torch.rand(k)
        alpha = torch.randn(k)
        a, T = make_ft(lambdas, ps, alpha, k)
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
        a, T = make_ft(lambdas, ps, alpha, k)

        # External moment computation
        m_there = compute_first_n_moments(a, T, n=2*k-1)

        # This moment computation
        m_here = compute_moments(a, T, k, 2*k-1)

        # Compare
        for i, (m1, m2) in enumerate(zip(m_here, m_there)):
            print(f"Moment {i+1} is {m1:.3f} and {m2:.3f}")

    make_a_ft()
    compare_moment_methods()

    ms = torch.tensor([4.438, 38.640, 502.534, 8705.890, 188486.062], dtype=torch.float32)
    fit_ft_distribution(ms, 3, num_epochs=200000)
