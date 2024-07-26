
import numpy as np
from scipy import stats


def sample_one_of_distributions(
    distributions: list[stats.rv_continuous],
    weights: list[float] = None
):
    """
    Sample from a list of distributions with given weights
    """
    if weights:
        assert len(distributions) == len(weights)
        assert sum(weights) == 1

    def rvs():
        # Choose a distribution
        dist = np.random.choice(distributions, p=weights if weights else None)
        return dist.rvs()

    class NewDist(stats.rv_continuous):
        def rvs(self):
            return rvs()

    return NewDist()