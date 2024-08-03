import math
import joblib

from daily.custom_poll_avg import estimate_margin_error
from election_statics import BIDEN, HARRIS
import matplotlib.pyplot as plt
import numpy as np
from harris.typical_variances import find_max_swing_row, average_swing_all_cycle
from hyperparams import swing_states
from simulate import estimate_fracs, simulate_election_mc, PollMissKind
from joblib import Memory
from pathlib import Path
cur_path = Path(__file__).parent.absolute()

cache = Memory(cur_path / "cache")

sim_count = 1000


def get_prob_for_averages():
    averages = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    moves = []
    for avg in averages:
        move = estimate_fracs(simulate_election_mc(
            #n_simulations=3000,
            n_simulations=sim_count,
            dem_candidate=BIDEN,
            poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
            average_movement=avg,
        ))[0]
        moves.append(move)
    return list(zip(averages, moves))


def get_prob_for_chaoses():
    averages = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    moves = []
    for avg in averages:
        move = estimate_fracs(simulate_election_mc(
            #n_simulations=3000,
            n_simulations=sim_count,
            dem_candidate=HARRIS,
            poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
            correlated_chaos_avg_change=avg,
        ))[0]
        moves.append(move)
    return list(zip(averages, moves))


def plot_average_move_to_win_prob(
    save_path=None,
    is_mobile=False,
):
    data = get_prob_for_averages()
    x, y = zip(*data)
    x = np.array(x) * 100
    y = np.array(y) * 100
    #plt.plot(
    #    x, y)
    # do it seaborn style
    import seaborn as sns
    sns.set()
    # Set the figure size
    plt.figure(figsize=(5.5, 4))
    sns.lineplot(
        x=x, y=y,
        color='orange', marker='o'
    )
    plt.xlabel("Average state polls movement (could be up or down)")
    plt.ylabel("Probability of Biden winning")
    plt.title("How average assumed poll movement affects Biden's chances")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 110, 10), [f"{i}%" for i in range(0, 110, 10)])
    annotations = [
        (average_swing_all_cycle(), "Average Move in '20 & '24"),
        (find_max_swing_row()['swing_abs'], "Largest Move in '20 & '24"),
        (np.mean([
            find_max_swing_row(state, 2020)['swing_abs']
            for state in swing_states
        ]), "Avg Largest Move in '20"),
    ]
    # Add vertical dashed line for annotations
    for x_val, label in annotations:
        plt.axvline(x=x_val, color='k', linestyle='--')
        plt.text(x_val + 0.2, 35, label, rotation=90)
    plt.xticks(np.arange(0, max(x) + 1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, transparent=True)
        plt.close()
        return save_path
    else:
        plt.show()


@cache.cache()
def find_mean_expected_error_poll(n, lower_bound=0.4, upper_bound=0.6, sample_scale=1000):
    """We want to estimate the expected error from a poll true value
    given the number of people polled"""
    # There is probably some analytical solution, but just going to estimate it
    average_miss = []
    for sim in range(sample_scale):
        true_value = np.random.uniform(lower_bound, upper_bound)
        num_surveys = sample_scale
        responses = np.random.binomial(p=true_value, n=n, size=num_surveys)
        estimates = responses / n
        misses = np.abs(estimates - true_value)
        average_miss.append(misses.mean())
    return np.mean(average_miss)


def plot_average_harris_delta_error(
    save_path=None,
    is_mobile=False,
):
    data = get_prob_for_chaoses()
    x, y = zip(*data)
    x = np.array(x) * 100
    y = np.array(y) * 100
    # do it seaborn style
    import seaborn as sns
    sns.set()
    # Set the figure size
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        x=x, y=y,
        color='pink', marker='o'
    )
    plt.xlabel("Average Harris Error (could be up or down) on Switch")
    plt.ylabel("Probability of Harris winning")
    plt.title("How mean absolute error on change affect Harris's chances")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 110, 10), [f"{i}%" for i in range(0, 110, 10)])
    annotations = [
        (
            find_mean_expected_error_poll(800) * 100,
            "Statistical Mean Absolute Error\nof an 800 person poll"
        ),
        (2, "Typical Miss From Polls-Avg\nNormal Candidate"),
        (4, "Largest Natl Polls-Avg Miss\nsince 1976"),
    ]
    # Add vertical dashed line for annotations
    for x_val, label in annotations:
        plt.axvline(x=x_val, color='k', linestyle='--')
        plt.text(x_val + 0.1, 35, label, rotation=90, fontsize=8)
    plt.xticks(np.arange(0, max(x) + 1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, transparent=True)
        plt.close()
        return save_path
    else:
        plt.show()


if __name__ == "__main__":
    #print(average_swing_all_cycle(HARRIS, "Florida"))
    #print(estimate_margin_error(1.7) / 2)
    plot_average_move_to_win_prob()
    #plot_average_harris_delta_error()