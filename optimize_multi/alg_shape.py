import torch
from alg import GeneralPHMatcher, CoxianPHMatcher, HyperErlangMatcher

THE_GLOBAL_TRADEOFF_PARAMETER = .005


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
        a_expTx = torch.bmm(a.unsqueeze(1).to(a.device), expTx)  # shape (n, 1, k)

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

        a = a.to(self.device)
        T = T.to(self.device)
        loss2 = 0

        for x, p in target_ms["cdf"]:
            cdf_at_x = self.phase_type_cdf(a, T, x)
            loss2 += torch.sum((cdf_at_x - p) ** 2)

            extended_loss_info["per_replica"] += THE_GLOBAL_TRADEOFF_PARAMETER * (cdf_at_x - p) ** 2

        return loss1 + THE_GLOBAL_TRADEOFF_PARAMETER * loss2, extended_loss_info


class CoxianPHPercentileMatcher(CoxianPHMatcher):
    def _loss(self, target_ms):
        """
        target_ms is a dictionary with keys "moments" and "cdf"
        """
        target_moments = target_ms["moments"]
        loss1, extended_loss_info = super()._loss(target_moments)

        a, T = self._make_phs_from_params()
        loss2 = 0

        for x, p in target_ms["cdf"]:
            cdf_at_x = GeneralPHPercentileMatcher.phase_type_cdf(a, T, x)
            loss2 += torch.sum((cdf_at_x - p) ** 2)

            extended_loss_info["per_replica"] += THE_GLOBAL_TRADEOFF_PARAMETER * (cdf_at_x - p) ** 2

        return loss1 + THE_GLOBAL_TRADEOFF_PARAMETER * loss2, extended_loss_info


class HyperErlangPHPercentileMatcher(HyperErlangMatcher):
    def _loss(self, target_ms):
        """
        target_ms is a dictionary with keys "moments" and "cdf"
        """
        target_moments = target_ms["moments"]
        target_moments = target_moments.to(self.device)
        loss1, extended_loss_info = super()._loss(target_moments)

        a, T = self._make_phs_from_params()
        a = a.to(self.device)
        loss2 = 0

        for x, p in target_ms["cdf"]:
            cdf_at_x = GeneralPHPercentileMatcher.phase_type_cdf(a, T, x)
            loss2 += torch.sum((cdf_at_x - p) ** 2)

            extended_loss_info["per_replica"] += THE_GLOBAL_TRADEOFF_PARAMETER * (cdf_at_x - p) ** 2

        return loss1 + THE_GLOBAL_TRADEOFF_PARAMETER * loss2, extended_loss_info


if __name__ == "__main__":
    # m = GeneralPHPercentileMatcher(ph_size=10, n_replica=5, num_epochs=1000)
    # m = CoxianPHPercentileMatcher(ph_size=10, n_replica=5, num_epochs=1000)
    m = HyperErlangPHPercentileMatcher(block_sizes=[14, 7, 21, 8], n_replica=1500, num_epochs=20000)

    moms = torch.tensor([1. , 1.30085805, 1.99965499, 3.47521126, 6.68631129])

 #    cdf = {0.31781939799331105: 0.10466718180975276,
 # 0.4683180602006689: 0.20123510098907926,
 # 0.6522608695652173: 0.3040068313411708,
 # 0.8027595317725753: 0.4054009223876951,
 # 0.9532581939799332: 0.5036442460985209,
 # 1.1204789297658864: 0.606067793205803,
 # 1.2876996655518396: 0.7054853565957587,
 # 1.4716424749163879: 0.8022630902992995,
 # 1.7391956521739131: 0.9018651072360979}

 #    cdf = {0.25093110367892973: 0.05748600405483539,
 # 0.31781939799331105: 0.10466718180975276,
 # 0.3847076923076923: 0.15138501855144215,
 # 0.4683180602006689: 0.20123510098907926,
 # 0.5686505016722408: 0.2546819549566005,
 # 0.6522608695652173: 0.3040068313411708,
 # 0.7358712374581939: 0.3595194335729923,
 # 0.8027595317725753: 0.4054009223876951,
 # 0.8696478260869566: 0.45018649220208484,
 # 0.9532581939799332: 0.5036442460985209,
 # 1.0368685618729097: 0.5551975929807758,
 # 1.1204789297658864: 0.606067793205803,
 # 1.2040892976588629: 0.6564495583798595,
 # 1.2876996655518396: 0.7054853565957587,
 # 1.371310033444816: 0.7518526867409857,
 # 1.4716424749163879: 0.8022630902992995,
 # 1.5886969899665553: 0.8522366138616284,
 # 1.7391956521739131: 0.9018651072360979,
 # 1.9733046822742475: 0.9510165433845462}


    cdf =  {0.31781939799331105: 0.10466718180975276,
 0.6522608695652173: 0.3040068313411708,
 0.9532581939799332: 0.5036442460985209,
 1.2876996655518396: 0.7054853565957587,
 1.7391956521739131: 0.9018651072360979}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cdf = cdf.to(device)
    # moms = moms.to(device)

    m.fit({"moments": moms, "cdf": list(cdf.items())})

    a, T = m._make_phs_from_params()
    a = a.to(m.device)
    print("="*10 + " Moments " + "="*10)
    print(m._compute_moments(a, T, n_moments=20, device=m.device))

    print("=" * 10 + " CDF " + "=" * 10)
    for x in cdf.keys():
        print(GeneralPHPercentileMatcher.phase_type_cdf(a, T, x))

    print("=" * 10 + " PH " + "=" * 10)

    a, T = m.get_best_after_fit()
    print(a)
    print(T)
    import pickle as pkl
    pkl.dump((a,T), open('a_T_hyper_5.pkl', 'wb'))

