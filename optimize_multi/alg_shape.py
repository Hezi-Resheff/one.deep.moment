import torch
from alg import GeneralPHMatcher

THE_GLOBAL_TRADEOFF_PARAMETER = .1


class GeneralPHPercentileMatcher(GeneralPHMatcher):

    @staticmethod
    def phase_type_cdf(a, T, x):
        """
        Compute the CDF of a phase-type distribution at x for n replicas of (a, T) using PyTorch.

        Args:
            a: torch.Tensor of shape (n, k), where each row is an initial probability vector.
            T: torch.Tensor of shape (n, k, k), where each (k, k) matrix is a sub-transition rate matrix.
            x: scalar, the point at which to evaluate the CDF.

        Returns:
            torch.Tensor of shape (n,), the CDF values for each replica.
        """
        n, k = a.shape
        ones = torch.ones(k, 1, device=a.device)

        expTx = torch.matrix_exp(T * x)  # shape (n, k, k)

        # Multiply a with expTx: (n, 1, k) @ (n, k, k) -> (n, 1, k)
        a_expTx = torch.bmm(a.unsqueeze(1), expTx)  # shape (n, 1, k)

        # Compute CDF
        cdf = 1 - torch.bmm(a_expTx, ones.unsqueeze(0).expand(n, -1, -1)).squeeze(-1).squeeze(-1)  # shape (n,)

        return cdf

    def _loss(self, target_ms):
        """
        target_ms is a dictionary with keys "moments" and "cdf"
        """
        target_moments = target_ms["moments"]
        loss1, extended_loss_info = super()._loss(target_moments)

        a, T = self._make_phs_from_params()
        loss2 = 0

        for x, p in target_ms["cdf"]:
            cdf_at_x = self.phase_type_cdf(a, T, x)
            loss2 += torch.sum((cdf_at_x - p) ** 2)

        return loss1 + THE_GLOBAL_TRADEOFF_PARAMETER * loss2, extended_loss_info


if __name__ == "__main__":
    m = GeneralPHPercentileMatcher(ph_size=10, n_replica=250, num_epochs=50000)

    moms = torch.tensor([1.00000000e+00, 1.79383841e+00, 4.46889373e+00, 1.40808072e+01, 5.34659152e+01])

    cdf = {0.3253253253253253: 0.25089067220687866,
           0.7657657657657657: 0.5022503137588501,
           1.4314314314314314: 0.7513095736503601}

    m.fit({"moments": moms, "cdf": list(cdf.items())})

    a, T = m._make_phs_from_params()

    print("="*10 + " Moments " + "="*10)
    print(m._compute_moments(a, T, n_moments=5, device=m.device))

    print("=" * 10 + " CDF " + "=" * 10)
    for x in cdf.keys():
        print(m.phase_type_cdf(a, T, x))

    print("=" * 10 + " PH " + "=" * 10)

    a, T = m.get_best_after_fit()
    print(a)
    print(T)

