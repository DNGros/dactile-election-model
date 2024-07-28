import functools
import random
import math
from collections import defaultdict
from typing import Literal
import numpy as np
from enum import StrEnum
from scipy import stats
from tqdm import tqdm

from daily.custom_poll_avg import get_sim_custom_averages, estimate_margin_error
from election_ops import ElectionScore
from election_statics import get_all_state_codes, BIDEN, WHITMER, HARRIS, state_code_to_state_name
import pandas as pd
from election_structs import Election, StateBeliefs, StateResult
from get_poll_data import get_state_averages
from harris.harris_explore import harris_swing_state_table_html
from harris.typical_variances import average_swing_all_cycle
from historical_elections import get_2020_election_struct
from hyperparams import default_poll_miss, poll_miss_div, poll_miss_for_other_mis_div, whitmer_mi_bump, \
    default_chaos_factor, adjusted_poll_miss, harris_national_change, \
    default_movement_cur_cycle_average_multiple, harris_article_calc_date, dropout_day
from state_correlations import apply_correlation, load_random_correlations, load_random_covariances, \
    get_random_multivariate_t_dist, load_538_covariances, corr2cov, load_five_thirty_eight_correlation, \
    calc_scale_factor_for_t_dist
from joblib import Memory
from pathlib import Path

cur_path = (Path(__file__).parent).absolute()

cache_dir = str(cur_path / "simcache")
cache = Memory(cache_dir, verbose=1)
#cache = Memory(None, verbose=1)


class PollMissKind(StrEnum):
    SIMPLE_MISS = "simple_miss"
    """Use the 538 typical poll miss"""
    ADJUSTED = "adjusted"
    """Use an increased poll miss for being longer from the election. About 2x miss"""
    RECENT_CYCLE = "recent_cycle"
    """Use a Dem/Rep margin-based miss from recent cycles"""
    RECENT_CYCLE_CORRELATED = "recent_cycle_correlated"
    """Use a Dem/Rep margin-based miss from recent cycles, but with state-level correlations"""
    POLL_MISS_TODAY_CORRELATED = "poll_miss_today"


def get_nearest_date(df, target_date, return_as_string=True):
    """
    Finds the nearest date in the DataFrame to the target_date.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'date' column with datetime objects.
    target_date (str): The target date as a string in the format 'YYYY-MM-DD'.
    return_as_string (bool): Whether to return the nearest date as a string.

    Returns:
    pd.Timestamp or str: The nearest date in the DataFrame to the target_date.
    """
    target_date = pd.to_datetime(target_date)
    date_series = pd.to_datetime(df['date'])  # Create a copy of the 'date' column as datetime
    nearest_date = date_series.iloc[(date_series - target_date).abs().argsort()[:1]].values[0]

    # Convert numpy.datetime64 to pd.Timestamp
    nearest_date = pd.Timestamp(nearest_date)

    if return_as_string:
        return nearest_date.strftime('%Y-%m-%d')
    return nearest_date


def get_dem_bump_for_candidate(
    candidate: str,
    state: str,
    correlation_power: float = 1.0
):
    if candidate == BIDEN:
        return 0
    elif candidate == WHITMER:
        bumps = apply_correlation("MI", value=whitmer_mi_bump.rvs(), correlation_power=correlation_power)
        return bumps[state]
    elif candidate == HARRIS:
        #_, state_changes = harris_swing_state_table_html()
        #return state_changes.get(state, state_changes[None])
        return 0
    else:
        raise ValueError

#NOTE to self.
# https://math.stackexchange.com/questions/555831/the-expectation-of-absolute-value-of-random-variables
# Standard dev of absolute values is sqrt(2/pi) * std dev


def initialize_election(
    #date="2024-07-06",
    #date: str = harris_article_calc_date.strftime("%Y-%m-%d"),
    date: str = pd.Timestamp.now().strftime("%Y-%m-%d"),
    dem_candidate=BIDEN,
    chaos_mean=0.0,
    chaos_std_dev=0.0,
    correlation_power: float = 1.0,
    polls_source: Literal['538', 'custom'] = 'custom',
):
    state_beliefs: list[StateBeliefs] = []

    baseline = get_2020_election_struct()
    if dem_candidate != BIDEN:
        #chaos_adjust = default_chaos_factor.rvs()
        chaos_adjust = stats.norm(loc=chaos_mean, scale=chaos_std_dev).rvs()
    else:
        chaos_adjust = 0
    if polls_source == '538':
        state_polls = get_state_averages()
        for state_code, df in state_polls.items():
            nearest_date = get_nearest_date(df, date)
            date_df = df[df["date"] == nearest_date]
            dem_pct = date_df["DEM_pct_estimate"].mean()
            rep_pct = date_df["REP_pct_estimate"].mean()
            other_pct = 0
            total_pct = dem_pct + rep_pct + other_pct
            # Note the total does not add to 1 because of undecided(?)
            belief = StateBeliefs(
                state=state_code,
                frac_dem_avg=dem_pct / total_pct,
                frac_rep_avg=rep_pct / total_pct,
                frac_other_avg=other_pct / total_pct,
                total_votes=baseline.state_results[state_code].total_votes,
                dem_bump_from_new_candidate=get_dem_bump_for_candidate(
                    dem_candidate, state_code,
                    correlation_power=correlation_power,
                ),
                dem_chaos_factor=chaos_adjust,
                weighted_poll_count=None,
            )
            state_beliefs.append(belief)
    else:
        date_as_timestamp = pd.to_datetime(date)
        state_code_to_dem_frac = get_sim_custom_averages(
            date=date_as_timestamp,
            candidate=dem_candidate,
        )
        for state_code, state_data in state_code_to_dem_frac.items():
            belief = StateBeliefs(
                state=state_code,
                frac_dem_avg=state_data['average'],
                frac_rep_avg=1 - state_data['average'],
                frac_other_avg=0,
                total_votes=baseline.state_results[state_code].total_votes,
                dem_bump_from_new_candidate=get_dem_bump_for_candidate(
                    dem_candidate, state_code,
                    correlation_power=correlation_power,
                ),
                dem_chaos_factor=chaos_adjust,
                weighted_poll_count=state_data['total_weight_sum']
            )
            state_beliefs.append(belief)

    return Election({}, state_beliefs, dem_candidate=dem_candidate)


def belief_to_result(
    belief: StateBeliefs,
    dem_adjustment: float,
    poll_miss: PollMissKind = PollMissKind.SIMPLE_MISS,
    correlated_t_samples: tuple[float, ...] = None,
    average_movement: float = None,
    correlated_chaos_avg_change: float = 0,
) -> StateResult:
    """Gets a new result given the sampled changes"""
    if poll_miss in (
        PollMissKind.RECENT_CYCLE,
        PollMissKind.RECENT_CYCLE_CORRELATED,
        PollMissKind.POLL_MISS_TODAY_CORRELATED
    ):
        frac_dem = belief.frac_dem_avg
        frac_rep = belief.frac_rep_avg
        frac_other = 0  # Essentially assume this splits equally
        total = frac_dem + frac_rep
        frac_dem /= total
        frac_rep /= total
        dem_margin = frac_dem - frac_rep
        avg_margin_miss = 0.038  # From the 538 num https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=110867585
        if belief.weighted_poll_count is not None:
            # Adjust for how much polling we have. Less polling increases miss.
            avg_margin_miss = estimate_margin_error(
                cur_weight_sum=belief.weighted_poll_count,
                target_full_error=avg_margin_miss,
            )
        #print("average margin miss", avg_margin_miss)

        # Estimate expected movement scaling
        df = 5
        if average_movement is not None:
            pass
        elif poll_miss in (PollMissKind.RECENT_CYCLE, PollMissKind.RECENT_CYCLE_CORRELATED):
            average_movement = (
                average_swing_all_cycle() / 100
                * default_movement_cur_cycle_average_multiple
            )
        else:
            average_movement = 0
        #print("average movement", average_movement)

        if poll_miss == PollMissKind.RECENT_CYCLE:
            raise NotImplementedError
        else:
            margin_swing = (
                correlated_t_samples[0] * calc_scale_factor_for_t_dist(
                    df, avg_margin_miss
                )
            )
            dem_margin += margin_swing
            dem_margin = float(np.clip(dem_margin, -1, 1))
            frac_dem = (1 + dem_margin) / 2
            frac_dem += correlated_t_samples[1] * calc_scale_factor_for_t_dist(df, average_movement)
            frac_dem += correlated_t_samples[2] * calc_scale_factor_for_t_dist(df, correlated_chaos_avg_change)
            frac_dem += belief.dem_bump_from_new_candidate
            frac_dem = float(np.clip(frac_dem, 0, 1))
        frac_rep = 1 - frac_dem
        if belief.dem_chaos_factor != 0:
            raise NotImplementedError
    elif poll_miss in (PollMissKind.SIMPLE_MISS, PollMissKind.ADJUSTED):
        # DEPRECATED
        if correlated_chaos_avg_change != 0:
            raise NotImplementedError
        dist = default_poll_miss if poll_miss == "default" else adjusted_poll_miss
        dem_error = dist.rvs() / poll_miss_div
        rep_error = dist.rvs() / poll_miss_div

        # Apply the errors to the average fractions
        frac_dem = belief.frac_dem_avg + dem_error
        frac_rep = belief.frac_rep_avg + rep_error
        frac_other = belief.frac_other_avg + (default_poll_miss.rvs() / poll_miss_for_other_mis_div / 2)
        # Renormalize the fractions so that their sum is 1
        total = frac_dem + frac_rep
        frac_dem /= total
        frac_rep /= total
        # frac_other /= total
        frac_other = 0

        dem_adjust = belief.dem_bump_from_new_candidate + belief.dem_chaos_factor
        frac_dem += dem_adjust
        frac_rep -= dem_adjust

        frac_dem = float(np.clip(frac_dem, 0, 1))
        frac_rep = float(np.clip(frac_rep, 0, 1))

        total = frac_dem + frac_rep
        frac_dem /= total
        frac_rep /= total
    else:
        raise ValueError

    # Try just make the swing states random
    #frac_dem = 1 if random.random() > 0.5 else 0
    #frac_rep = 1 - frac_dem

    # Determine the winner
    winner = "DEM" if frac_dem > frac_rep else "REP"

    return StateResult(
        state=belief.state,
        winner=winner,
        frac_dem=frac_dem,
        frac_rep=frac_rep,
        frac_other=frac_other,
        total_votes=belief.total_votes,
        from_beliefs=belief,
    )


def simulate_election_once(
    election: Election, poll_miss, average_movement,
    correlated_chaos_avg_change: float = 0,
):
    state_results = election.state_results
    need_correlated_samples = poll_miss in (
        PollMissKind.RECENT_CYCLE_CORRELATED,
        PollMissKind.POLL_MISS_TODAY_CORRELATED
    )
    if need_correlated_samples:
        #samples = stats.multivariate_normal.rvs(mean=np.zeros(len(states)),
        #                                        cov=cov_matrix,
        #                                        size=num_of_rand_samples_needed)
        #samples = dist.rvs(size=num_of_rand_samples_needed)

        #vals = []

        num_of_rand_samples_needed = 4
        dist, states = get_random_multivariate_t_dist(
            degrees_freedom=5,
            states=tuple([b.state for b in election.remaining_states])
        )
        samples = dist.rvs(size=num_of_rand_samples_needed)
        state_to_samples = {state: samples[:, i] for i, state in enumerate(states)}
        #print(state_to_samples)
    for belief in election.remaining_states:
        #print("State", belief.state)
        state_results[belief.state] = belief_to_result(
            belief,
            dem_adjustment=0,
            poll_miss=poll_miss,
            correlated_t_samples=tuple(state_to_samples.get(belief.state)) if need_correlated_samples else None,
            average_movement=
                average_movement if average_movement is not None else (
                    average_swing_all_cycle(state=state_code_to_state_name(belief.state), candidate=election.dem_candidate) / 100
                ),
            correlated_chaos_avg_change=correlated_chaos_avg_change,
        )
    return Election(state_results, [], dem_candidate=election.dem_candidate)


@cache.cache()
def simulate_election_mc(
    n_simulations: int = 10_000,
    dem_candidate: str = BIDEN,
    poll_miss: PollMissKind = PollMissKind.RECENT_CYCLE_CORRELATED,
    chaos_dem_mean: float = 0,
    chaos_std_dev: float = 0.0,
    correlation_power: float = 1.0,
    correlated_chaos_avg_change: float = 0,
    average_movement: float = None,
    poll_source: Literal['538', 'custom'] = 'custom',
    reference_today_date: pd.Timestamp = pd.Timestamp.now().normalize(),
) -> list[ElectionScore]:
    """The starting point of monte carlo simulations of the election"""
    scores = []
    baseline = get_2020_election_struct()
    for _ in tqdm(range(n_simulations)):
        election = initialize_election(
            dem_candidate=dem_candidate,
            chaos_mean=chaos_dem_mean,
            chaos_std_dev=chaos_std_dev,
            correlation_power=correlation_power,
            polls_source=poll_source,
            date=reference_today_date.strftime("%Y-%m-%d"),
        )
        new_election = simulate_election_once(
            election,
            poll_miss=poll_miss,
            average_movement=average_movement,
            correlated_chaos_avg_change=correlated_chaos_avg_change,
        )
        scores.append(ElectionScore.from_election(new_election, baseline))
    return scores


def average_poll_miss(
    poll_miss,
    n_simulations=10_000,
):
    """Check that the empirically measured poll miss is close
    to what we expect. Note we are doing this on the average miss of both
    the D and R fraction after removing 3rd party votes. It's not clear
    if this is as expected."""
    samples = simulate_election_mc(
        n_simulations=n_simulations,
        dem_candidate=HARRIS,
        poll_miss=poll_miss,
    )
    misses = []
    margin_misses = []
    state_to_margin_misses = defaultdict(list)
    for sample in samples:
        for state_code, result in sample.election.state_results.items():
            dem_miss = result.frac_dem - result.from_beliefs.frac_dem_avg
            #print("Result frac_dem:", result.frac_dem, "Belief frac_dem_avg:", result.from_beliefs.frac_dem_avg)
            rep_miss = result.frac_rep - result.from_beliefs.frac_rep_avg
            margin_miss = (
                (result.frac_dem - result.frac_rep)
                - (result.from_beliefs.frac_dem_avg - result.from_beliefs.frac_rep_avg)
            )
            misses.append(abs(dem_miss))
            misses.append(abs(rep_miss))
            margin_misses.append(abs(margin_miss))
            state_to_margin_misses[state_code].append(abs(margin_miss))
    average_miss = sum(misses) / len(misses)
    print("Average poll miss:", average_miss)
    print("Average margin miss:", sum(margin_misses) / len(margin_misses))
    #print("Average margin swing super total:", sum(_all_margin_swings) / len(_all_margin_swings))
    for state_code, margin_misses in state_to_margin_misses.items():
        print(state_code, "mean", sum(margin_misses) / len(margin_misses))
    v, states = estimate_fracs(samples)
    print("Win rate:", v)
    return average_miss


def estimate_fracs(
    sims: list[ElectionScore],
):
    """Given a list of election results, will get a total win
    rate and a rate for each state"""
    win_counts = {"DEM": 0, "REP": 0}
    state_win_counts = {
        state_code: {"DEM": 0, "REP": 0}
        for state_code in get_all_state_codes()
    }
    for sim in sims:
        win_counts[sim.winner] += 1
        for state_code in get_all_state_codes():
            state_win_counts[state_code][sim.state_to_winner[state_code]] += 1

    dem_overall_frac = win_counts["DEM"] / (win_counts["DEM"] + win_counts["REP"])
    dem_win_fracs = {
        state: counts["DEM"] / (counts["DEM"] + counts["REP"])
        for state, counts in state_win_counts.items()
    }
    #plot_election_map(dem_win_fracs)
    return dem_overall_frac, dem_win_fracs


if __name__ == "__main__":
    overall, states = estimate_fracs(simulate_election_mc(
        #dem_candidate=BIDEN,
        dem_candidate=HARRIS,
        poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        #reference_today_date=dropout_day,
    ))
    print(overall)
    for state_code, frac in states.items():
        print(state_code, frac)
