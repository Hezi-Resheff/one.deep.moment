"""
Can we match 20 moments??
"""
from matching import *
from util import *


if __name__ == "__main__":
    orig_size = 50   # This is the size of the PH the moments come from (so we know they are feasible)
    use_size = 50    # This is the size of the target PH
              # This is the number of moments to match

    n = 20

    ms = get_feasible_moments(original_size=orig_size, n=n)
    print(ms)
    num_epochs = 100000
    ws = ms ** (-1)

    matcher = MomentMatcher(ms)
    (lambdas, ps, alpha), (a, T) = matcher.fit_ph_distribution(use_size, num_epochs=num_epochs, moment_weights=ws)

    moment_table = moment_analytics(ms, compute_moments(a, T, use_size, n))
    print(moment_table)


