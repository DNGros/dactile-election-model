import dataclasses
from tqdm import tqdm

import numba
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import Literal, Callable

from election_statics import DEM, REP, OTHER
from joblib import Memory

cache = Memory("cache", verbose=1)


@dataclasses.dataclass
class Voter:
    vote: Literal[DEM, REP, OTHER] = None


class DecisionPoint:
    pass


class CategoricalDecisionPoint:
    def __init__(self, name: str, options: list[str]):
        pass

from bisect import bisect_right
import random
import numpy as np

def weighted_shuffle(v):
    a, w = zip(*v)
    r = np.empty_like(a)
    cumWeights = np.cumsum(w)
    for i in range(len(a)):
         rnd = random.random() * cumWeights[-1]
         j = bisect_right(cumWeights,rnd)
         r[i]=a[j]
         cumWeights[j:] -= w[j]
    return r


def pred_voter(
    voter: Voter,
    variables: dict[str, bool | int | float | str],
    opinions: dict[str, list[float]],
):
    #issues = list(weighted_shuffle([
    #    ('economy', 1),
    #    ('fitness', 1),
    #    ('personal_conduct', 1),
    #    ('issues', 1),
    #    #('identity', 1),
    #]))
    ## reverse issues (since weighting is backwards)
    #issues = issues[::-1]
    issues = [
        'economy',
        'fitness',
        'personal_conduct',
        'issues',
    ]
    random.shuffle(issues)

    split_third_party_prob = 0.5
    #split_third_party_prob = 0
    lean_prob = 0.5
    #lean_prob = 0.7
    while issues and voter.vote is None:
        issue = issues.pop()
        if issue == 'fitness':
            fit = random.choice([
                'mentally_sharp',
                'cares_about_ordinary_people',
                'honest',
            ])
            voter.vote = sample_from_4_way(
                opinions[fit],
                #[0.04, 0.2, 0.24, 0.32],
                #[0.34, 0.2, 0.24, 0.32],
                #[0.23, 0.26, 0.2, 0.34],  # Cares about ordinary people
                #[0.2, 0.28, 0.24, 0.13],  # Honest
                #[0.13, 0.28, 0.24, 0.13],  # Honest but less
                lean_prob,
                split_third_party_prob,
            )
        elif issue == 'personal_conduct':
            # Pew https://www.pewresearch.org/politics/2024/07/11/biden-and-trumps-personal-qualities-and-handling-of-issues/
            d = opinions['personal_conduct_d']
            r = opinions['personal_conduct_r']
            like, mixed, dislike = 0, 1, 2
            d_skew = d[like] + r[dislike]
            r_skew = d[dislike] + r[like]
            mixed_skew = d[mixed] + r[mixed]
            choice = random.choices(["d", "mixed", "r"], [d_skew, mixed_skew, r_skew])[0]
            if choice == "d":
                voter.vote = DEM
            elif choice == "r":
                voter.vote = REP
            else:
                if random.random() < split_third_party_prob:
                    voter.vote = OTHER
        elif issue == 'economy':
            # Pew poll https://www.pewresearch.org/politics/2024/07/11/biden-and-trumps-personal-qualities-and-handling-of-issues/
            voter.vote = sample_from_4_way(
                opinions['economy'],
                lean_prob,
                split_third_party_prob
            )
        elif issue == 'issues':
            do_issues_vote(voter, lean_prob, split_third_party_prob, opinions)
        elif issue == 'identity':
            do_identity_vote(voter, lean_prob, split_third_party_prob)
        else:
            raise ValueError
    if voter.vote is None:
        if split_third_party_prob > 0:
            voter.vote = random.choice([DEM, REP, OTHER])
        else:
            voter.vote = random.choice([DEM, REP])


def do_identity_vote(voter: Voter, lean_prob, split_third_party_prob):
    #voter.vote = sample_from_4_way(
    #    [0.15, 0.15, 0.15, 0.15],
    #    lean_prob,
    #    split_third_party_prob,
    #)
    pass


def do_issues_vote(voter: Voter, lean_prob, split_third_party_prob, opinions):
    sub_issue = random.choice([
        'foreign_policy',
        'good_government',
        'abortion',
        'supreme_court',
        'immigration',
    ])
    voter.vote = sample_from_4_way(
        opinions[sub_issue],
        lean_prob,
        split_third_party_prob,
    )


def sample_from_dist_very_somewhat(
    d_very_somewhat: list[float],
    r_very_somewhat: list[float],
    split_third_party_prob: float,
):
    dem = random.choices([0, 1], d_very_somewhat)[0]
    rep = random.choices([0, 1], r_very_somewhat)[0]
    both_like = dem == 0 and rep == 0
    both_dislike = dem == 1 and rep == 1
    if both_like or both_dislike:
        if random.random() < split_third_party_prob:
            return OTHER
        return None
    if dem == 0:
        return DEM
    if rep == 0:
        return REP

@numba.njit
def random_choice(choices, probs):
    probs_array = np.array(probs)
    cum_probs = np.cumsum(probs_array)
    rand_num = np.random.random()

    for i in range(len(cum_probs)):
        if rand_num < cum_probs[i]:
            return choices[i]
    return choices[-1]


@numba.njit
def sample_from_4_way(
    d_to_r: list[float],
    lean_prob: float,
    split_third_party_prob: float,
):
    assert len(d_to_r) == 4
    d_to_r = list(d_to_r)
    #d_to_r[0] *= 0.85
    #d_to_r[1] *= 0.9
    d_to_r.append(1 - sum(d_to_r))
    choice = random_choice(["d", "lean_d", "lean_r", "r", "skip"], d_to_r)
    if choice == "skip":
        return None
    elif choice == "d":
        return DEM
    elif choice == "r":
        return REP
    elif choice == "lean_d":
        if random.random() > lean_prob:
            return DEM
        if random.random() < split_third_party_prob:
            return OTHER
    elif choice == "lean_r":
        if random.random() > lean_prob:
            return REP
        if random.random() < split_third_party_prob:
            return OTHER
    else:
        raise ValueError(f"choice {choice} not found")
    return None


def distribution_to_dirichlet(dist: list[float], pseudocount: float) -> Callable[[], list[float]]:
    if sum(dist) > 1:
        dist = [x / sum(dist) for x in dist]
    counts = [
        pseudocount * x + 1e-6
        for x in dist
    ]
    counts += [pseudocount * (1 - sum(dist)) + 1e-6]

    def sampler():
        vals = np.random.dirichlet(counts)
        return vals[:-1]

    return sampler


def run_sim(voters: list[Voter], opinions):
    variables = {
        "is_economy_good": True,
        "dem_is_incumbent": 1,
    }
    for voter in voters:
        pred_voter(voter, variables, opinions)
    return voters


def voters_to_df(voters: list[Voter]):
    return pd.DataFrame([
        dataclasses.asdict(voter)
        for voter in voters
    ])


def get_silver_d_margein_to_win_prob():
    # https://www.natesilver.net/p/biden-needs-to-do-more-than-make
    return [
        (6.5 / 100, 0.9989744),
        (5.5 / 100, 0.9976452),
        (4.5 / 100, 0.9762578),
        (3.5 / 100, 0.893043),
        (2.5 / 100, 0.6933594),
        (1.5 / 100, 0.3801625),
        (0.5 / 100, 0.1445157),
        (-0.5 / 100, 0.0292363),
        (-1.5 / 100, 0.0057057),
        (-2.5 / 100, 0.0006371),
        (-3.5 / 100, 0.000357),
        (-4.5 / 100, 0),
        (-5.5 / 100, 0),
        (-6.5 / 100, 0),
    ]


def get_silver_func():
    x, y = zip(*get_silver_d_margein_to_win_prob())
    # Fit a sigmoid to the data
    from scipy.optimize import curve_fit
    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return y
    p0 = [1, 0, 1, 0]  # this is a mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x, y, p0, method='dogbox')
    print(popt)
    return lambda x: sigmoid(x, *popt)


def plot_silver():
    silver_func = get_silver_func()
    silver_data = get_silver_d_margein_to_win_prob()
    # Plot the silver func and data
    x = np.linspace(-7, 7, 100)
    y = silver_func(x)
    plt.plot(x, y)
    x, y = zip(*silver_data)
    plt.scatter(x, y)
    plt.show()


@cache.cache
def run_scenarios(opinions, num_samples):
    opinions_dirchlet = {
        #key: distribution_to_dirichlet(dist, 75)
        key: distribution_to_dirichlet(dist, 75 / 4)
        for key, dist in opinions.items()
    }

    def sample_opinions():
        return {
            key: d()
            for key, d in opinions_dirchlet.items()
        }

    dem_fracs = []
    rep_fracs = []
    other_fracs = []
    dem_adjusted_margins = []
    d_wins = []
    scenario_opinions = []

    dem_pop_vote_margin_to_win_prob = get_silver_func()
    for _ in tqdm(range(num_samples)):
        if num_samples > 1:
            op = sample_opinions()
        else:
            op = opinions
        voters = run_sim([Voter() for _ in range(10000)], op)
        df = voters_to_df(voters)
        votes = df['vote'].value_counts()
        dem_frac = votes.get(DEM, 0) / len(voters)
        dem_fracs.append(dem_frac)
        rep_frac = votes.get(REP, 0) / len(voters)
        rep_fracs.append(rep_frac)
        other_frac = votes.get(OTHER, 0) / len(voters)
        other_fracs.append(other_frac)

        adjusted_dem = dem_frac
        adjusted_rep = max(rep_frac - 0.01, 0)
        adjusted_other = other_frac / 2
        adjusted_sum = adjusted_dem + adjusted_rep + adjusted_other
        adjusted_dem /= adjusted_sum
        adjusted_rep /= adjusted_sum
        adjusted_other /= adjusted_sum
        d_margin = adjusted_dem - adjusted_rep
        dem_adjusted_margins.append(d_margin)
        win_prob = dem_pop_vote_margin_to_win_prob(d_margin)
        #print(f"{adjusted_dem=} {adjusted_rep=} {d_margin=} {win_prob=} {adjusted_sum=}")
        #exit()
        d_win = random.random() < win_prob
        d_wins.append(d_win)
        scenario_opinions.append(op)
        if num_samples == 1:
            print(votes)
            print("Dem frac:", dem_frac)
    # Make a dataframe
    df = pd.DataFrame({
        "dem_frac": dem_fracs,
        "rep_frac": rep_fracs,
        "other_frac": other_fracs,
        "dem_margin": dem_adjusted_margins,
        "d_win": d_wins,
        "opinions": scenario_opinions,
    })
    return df


def main():
    #plot_silver()
    #exit()
    # make a dirchlet distribution for the issues
    #all_dists = []
    #for dists in range(5000):
    #    dist = np.random.dirichlet([75/4/4, 75/4/4, 75/4/4, 75/4/4])
    #    all_dists.append({
    #        "A": dist[0],
    #        "B": dist[1],
    #        "C": dist[2],
    #    })
    ## plot box and whisker plot
    #df = pd.DataFrame(all_dists)
    #df.boxplot()
    #plt.show()
    ## std dev of a
    #print(np.std(df['A']))
    #exit()
    ## 90% interval on A
    #a = df["A"]
    #a = a.sort_values()
    #mean_a = a.mean()
    #print("a p05", a.quantile(0.05) - mean_a)
    #print("a p95", a.quantile(0.95) - mean_a)
    ## average miss
    #exit()

    opinions = {
        # Fit
        "mentally_sharp": [0.04, 0.2, 0.24, 0.32],
        #"mentally_sharp": [0.32, 0.24, 0.24, 0.32],
        "cares_about_ordinary_people": [0.23, 0.26, 0.2, 0.34],
        "honest": [0.2, 0.28, 0.24, 0.13],
        # Conduct
        "personal_conduct_d": [0.26, 0.3, 0.43],
        "personal_conduct_r": [0.14, 0.3, 0.56],
        # Econ
        'economy': [0.16, 0.24, 0.2, 0.29],
        # Issues
        'good_government': [0.11, 0.29, 0.24, 0.08],
        'immigration': [0.12, 0.24, 0.2, 0.34],
        'abortion': [0.24, 0.24, 0.25, 0.19],
        'foreign_policy': [0.16, 0.23, 0.2, 0.29],
        'supreme_court': [0.24, 0.23, 0.21, 0.26],
    }

    def hit_dems(
        opinions,
        very_hit: float,
        somewhat_hit: float,
        targeted_hit: dict[str, tuple[float, float]],
    ):
        """Adjusts down the very and somewhat frac for dems"""
        out = {}
        for key in opinions.keys():
            v = opinions[key]
            if len(v) != 4:
                out[key] = v
                continue
            use_very_hit, use_somewhat_hit = targeted_hit.get(key, (very_hit, somewhat_hit))
            out[key] = [v[0] * use_very_hit, v[1] * use_somewhat_hit, v[2], v[3]]

        return opinions

    #opinions = hit_dems(opinions, 0.85, 0.9, {'immigration': (0.7, 0.8)})

    df = run_scenarios(opinions, num_samples=1_000)
    print("Mean dem frac:", np.mean(df['dem_frac']))
    print("Std dev dem frac:", np.std(df['dem_frac']))
    print("Mean rep frac:", np.mean(df['rep_frac']))
    print("Std dev rep frac:", np.std(df['rep_frac']))
    print("Mean margin:", np.mean(df['dem_margin']))
    print("d win prob:", sum(df['d_win']) / len(df['d_win']))
    # plot the distribution of dem_fracs
    plt.hist(df['dem_frac'], bins=20, range=(0.3, 0.7))
    plt.show()
    # plot the margins
    plt.hist(df['rep_frac'], bins=20, range=(-0.1, 0.1))
    plt.title('Adjusted Dem margins')
    plt.show()

    # Filter to 2 < dem_margin < 3
    d_acceptable_margin_df = df[(df['dem_margin'] > 0.02) & (df['dem_margin'] < 0.03)]
    print(d_acceptable_margin_df)
    for opinion_cat in opinions.keys():
        average = np.mean([op[opinion_cat] for op in d_acceptable_margin_df['opinions']], axis=0)
        print(f"Average winning opinion for {opinion_cat}: {average}")
    print("Done")


if __name__ == "__main__":
    main()