import math
from typing import Literal
import functools
import pandas as pd
from election_statics import BIDEN, HARRIS, convert_state_name_to_state_code
from get_poll_data import get_poll_averages_from_csv
from harris.harris_explore import get_avail_filtered, poll_weight, build_polls_clean_df
from harris.margins import get_margins_df
from hyperparams import polling_measurable_error_frac, default_scale_national_for_states, dropout_day, \
    swing_states
from state_correlations import calc_scale_factor_for_t_dist
from util import pd_all_rows, pd_all_columns
from scipy import stats
from pathlib import Path

cur_path = Path(__file__).parent.absolute()

from joblib import Memory

cache = Memory(cur_path / "../cache")


@functools.cache
@cache.cache
def get_custom_polling_average(
    candidate: Literal["BIDEN", "HARRIS"],
    cycle: int = 2020,
    state: str | None = None,
    include_national: bool = True,
):
    df = build_polls_clean_df(
        mode=candidate,
        start_date=pd.Timestamp(f'{cycle}-02-01'),
        cycle=cycle,
        state=state,
    )
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #    print(df)
    #    print("suposed sum", df.custom_weight.sum())
    need_national_data = state is not None and include_national
    df = df[df['end_date'] < pd.Timestamp(f"{cycle}-11-05")]
    df['after_dropout'] = df['start_date'] > dropout_day
    if candidate == BIDEN:
        # Make sure not after dropout
        df = df[~df['after_dropout']]
    earliest_day = df['end_date'].min()
    today = pd.Timestamp.now()
    averages = []

    if need_national_data:
        national_averages = get_custom_polling_average(
            candidate, cycle, state=None, include_national=True).copy()
        slope, intercept, r_value, p_value, std_err = fit_linear_func_national_to_state(state)
        #                 ^ More correlated states get more weight
        #                   (there isn't really a principled reason for scaling though)
        national_averages['average'] = slope * national_averages['average'] + intercept
    else:
        national_weight = 0.0
    for day in pd.date_range(earliest_day, df.end_date.max() if cycle == 2020 else today):
        if day.day_of_year == today.day_of_year:
            print("TODAY")
        before_day = df[df['end_date'] <= day]
        before_day = before_day[before_day['end_date'] >= day - pd.Timedelta(days=60)]
        if before_day.empty:
            continue
        before_day_weight = before_day.apply(lambda x: poll_weight(
            numeric_grade=x['numeric_grade'],
            is_projection=False,
            start_date=x['start_date'],
            end_date=x['end_date'],
            pollscore=x['pollscore'],
            sample_size=x['sample_size'],
            reference_today_date=day,
            #time_penalty=9,  Just do the default
            harris_and_before_dropout=(candidate == HARRIS and not x['after_dropout']),
        ), axis=1)
        #if candidate == HARRIS:
        #    # Multiply values before the dropout by 0.25
        #    before_day_weight *= before_day['after_dropout'].apply(lambda x: 0.25 if x else 1)
        row = 'harris_frac' if candidate == HARRIS else 'biden_frac'
        frac_weighted = before_day_weight * (before_day[row] * 100)
        if need_national_data:
            national_weight = r_value ** 2 * default_scale_national_for_states
            if day <= dropout_day and candidate == HARRIS:
                national_weight *= 0.25
            nat_value = national_averages[national_averages['date'] == day]['average'].values[0]
        else:
            nat_value = 0.0
        #before_day = before_day.copy()
        #before_day['weight'] = before_day_weight
        #before_day['frac_weighted'] = frac_weighted
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        #    print(before_day)
        local_weight_sum = before_day_weight.sum()
        average = (
            (frac_weighted.sum() + (nat_value * national_weight))
            / (local_weight_sum + national_weight)
        )
        # Push further from 50
        #average = 0.5 + (average - 0.5)
        averages.append({
            'date': day,
            'average': average,
            'candidate': candidate,
            'local_weight_sum': local_weight_sum,
            "total_weight_sum": local_weight_sum + national_weight,
            'state': state,
            'cycle': cycle,
        })
    averages = pd.DataFrame(averages)
    return averages


def fit_linear_func_national_to_state(state):
    national_average = pd.concat([
        get_custom_polling_average(BIDEN, cycle=2020),
        get_custom_polling_average(BIDEN, cycle=2024),
    ])
    actual_poll_averages = get_margins_df(only_swing_states=False)
    actual_poll_averages = actual_poll_averages[actual_poll_averages.state == state]
    national_average = pd.merge(national_average, actual_poll_averages, on='date', how='left')
    national_average = national_average.dropna(subset=['average', 'percent_of_d_r_share'])
    x = national_average['average']
    y = national_average['percent_of_d_r_share']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value, p_value, std_err


def plot_poll_avgs():
    cycle = 2020
    #for state in swing_states:
    #    print(state)
    #    print(fit_linear_func_national_to_state(state))
    #exit()
    #state = "Pennsylvania"
    all_errors = []
    for state in swing_states:
        candidate = BIDEN
        averages = get_custom_polling_average(
            candidate,
            cycle,
            state=state if state != "National" else None,
            include_national=True,
        )
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            actual_poll_averages = get_margins_df(only_swing_states=False)
            #print(actual_poll_averages.state.unique())
            # Number of nulls
            #print(actual_poll_averages.isna().sum())
            actual_poll_averages = actual_poll_averages[actual_poll_averages.state == state]
            #print("Actual")
            #print(actual_poll_averages)
            # join with actual poll averages
            averages = pd.merge(averages, actual_poll_averages, on='date', how='left')
            #print("joined")
            #print(averages)

        # Filter averages starting march 1st
        averages = averages[averages['date'] >= pd.Timestamp(f'{cycle}-04-01')]
        # Plot the averages
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        import seaborn as sns
        sns.set()
        fig, ax = plt.subplots()
        # plot average and 'percent_of_d_r_share'
        ax.plot(averages['date'], averages['average'], label='Average')
        ax.plot(averages['date'], averages['percent_of_d_r_share'], label='Actual')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        #plt.legend()
        plt.title(f"{candidate} {state} {cycle} Polling Average")
        ax.set_ylim(40, 60)
        plt.tight_layout()
        plt.show()
        # Calculate the mean squared error
        averages['error'] = ((averages['average']) - averages['percent_of_d_r_share'])**2
        print("Mean local weight sum:")
        print(averages['local_weight_sum'].mean())
        print("Root mean squared error:")
        print(np.sqrt(averages['error'].mean()))
        print("Mean squared error:")
        print(averages['error'].mean())
        # Plot the weight sum
        fig, ax = plt.subplots()
        ax.plot(averages['date'], np.sqrt(averages['total_weight_sum']), label='Local Weight Sum')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        # set title
        plt.title(f"{state} {cycle} Polling Weight Sum")
        # set x-axis label
        plt.xlabel("Date")
        # set y-axis label
        plt.ylabel("Sqrt Weight Sum")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        all_errors.append(averages['error'].mean())
    print("Average error all")
    print(sum(all_errors) / len(all_errors))


@functools.cache
def find_weight_sum_for_day(
    date: pd.Timestamp,
    candidate: Literal["BIDEN", "HARRIS"],
    state: str | None,
):
    return _get_col_for_day(date, candidate, state, 'total_weight_sum')


def _get_col_for_day(
    date: pd.Timestamp,
    candidate: Literal["BIDEN", "HARRIS"],
    state: str | None,
    col: str,
):
    averages = get_custom_polling_average(
        candidate,
        cycle=int(date.year),
        state=state if state != "National" else None,
        include_national=True,
    )
    date = date.normalize()
    row = averages[averages['date'].dt.normalize() == date]
    if row.empty:
        return None
    return row[col].values[0]


def find_average_for_day(
    date: pd.Timestamp,
    candidate: Literal["BIDEN", "HARRIS"],
    state: str | None,
):
    return _get_col_for_day(date, candidate, state, 'average')


def find_national_avg_scaled_for_state(
    state: str | None,
    date: pd.Timestamp,
    candidate: Literal["BIDEN", "HARRIS"],
):
    nat_avg = find_average_for_day(date, candidate, state=None)
    slope, intercept, r_value, p_value, std_err = fit_linear_func_national_to_state(state)
    weight = r_value ** 2 * default_scale_national_for_states
    if date <= dropout_day and candidate == HARRIS:
        weight *= 0.25
    return slope * nat_avg + intercept, slope, intercept, weight


def find_election_day_2020_weight_sum(
    state=None
):
    return find_weight_sum_for_day(pd.Timestamp("2020-11-02"), candidate=BIDEN, state=state)


def find_biden_dropout_day(
    state=None
):
    return find_weight_sum_for_day(dropout_day, candidate=BIDEN, state=state)


@functools.cache
def get_all_swing_state_on_election_day_weight_counts():
    return [
        find_election_day_2020_weight_sum(state)
        for state in swing_states
    ]


def estimate_margin_error(
    cur_weight_sum,
    target_full_error: float = 0.038
):
    """
    Give some weighted count of polls, we want to estimate
    our average polling share margin absolute error.
    We assume that 2020 had a typical amount of weighted poll
    count, and tune a scaling based off that.
    """
    all_swing_states_weights = get_all_swing_state_on_election_day_weight_counts()
    # We know SD_mean = 1/n * sqrt(n*SD^2) if we had
    # These aren't actual normal distriubtions. But we'll
    # just solve for some constant
    # error = 1 / n * sqrt(n * E)
    # error * n = sqrt(n * E)
    # error^2 * n^2 = n * E
    # error^2 * n = E
    # Also we are going to assume that only a fraction of the error is can actually
    # be fixed with more polling
    targeted_fixable = target_full_error * polling_measurable_error_frac

    # We also don't quite want a simple average here of the weights.
    # Look chatgpt just magically solved this...
    """
    To solve for \( E \) in the equation

    \[
    \frac{\sum_{i=1}^n \frac{\sqrt{n_i \cdot E}}{n_i}}{n} = y
    \]

    we can follow these steps:

    1. Start with the given equation:
       \[
       \frac{\sum_{i=1}^n \frac{\sqrt{n_i \cdot E}}{n_i}}{n} = y
       \]

    2. Multiply both sides by \( n \) to eliminate the fraction:
       \[
       \sum_{i=1}^n \frac{\sqrt{n_i \cdot E}}{n_i} = y \cdot n
       \]

    3. Simplify the left side:
       \[
       \sum_{i=1}^n \frac{\sqrt{n_i \cdot E}}{n_i} = y \cdot n
       \]

    4. Recognize that each term \(\frac{\sqrt{n_i \cdot E}}{n_i}\) can be rewritten as \(\frac{\sqrt{n_i} \cdot \sqrt{E}}{n_i} = \frac{1}{\sqrt{n_i}} \cdot \sqrt{E}\):

       \[
       \sum_{i=1}^n \frac{\sqrt{E}}{\sqrt{n_i}} = y \cdot n
       \]

    5. Factor out \(\sqrt{E}\) from the sum:
       \[
       \sqrt{E} \sum_{i=1}^n \frac{1}{\sqrt{n_i}} = y \cdot n
       \]

    6. Solve for \(\sqrt{E}\) by dividing both sides by \(\sum_{i=1}^n \frac{1}{\sqrt{n_i}}\):
       \[
       \sqrt{E} = \frac{y \cdot n}{\sum_{i=1}^n \frac{1}{\sqrt{n_i}}}
       \]

    7. Square both sides to solve for \(E\):
       \[
       E = \left( \frac{y \cdot n}{\sum_{i=1}^n \frac{1}{\sqrt{n_i}}} \right)^2
       \]

    Therefore, the solution for \( E \) is:

    \[
    E = \left( \frac{y \cdot n}{\sum_{i=1}^n \frac{1}{\sqrt{n_i}}} \right)^2
    \]
    """

    n = len(all_swing_states_weights)
    y = targeted_fixable
    sum_1_over_sqrt_n_i = sum([
        1 / math.sqrt(n_i)
        for n_i in all_swing_states_weights
    ])
    E = (y * n / sum_1_over_sqrt_n_i) ** 2
    #print("E", E)
    return (
        math.sqrt(cur_weight_sum * E) / cur_weight_sum
        + (target_full_error * (1 - polling_measurable_error_frac))
    )


@functools.cache
def get_sim_custom_averages(
    date: pd.Timestamp,
    candidate: Literal["BIDEN", "HARRIS"] = HARRIS,
) -> dict[str, dict[str, float]]:
    states = [
        'Pennsylvania', 'Wisconsin', 'Georgia',
        'Michigan', 'North Carolina', 'Arizona', 'Nevada',
        'Florida'
    ]
    state_to_avg = {}
    for state in states:
        state_to_avg[convert_state_name_to_state_code(state)] = {
            'average': find_average_for_day(date, candidate, state) / 100,
            'total_weight_sum': find_weight_sum_for_day(date, candidate, state),
        }
    return state_to_avg


def average_estimated_miss():
    all_estimates = {}
    for state in swing_states:
        weight_sum = find_weight_sum_for_day(
            date=pd.Timestamp.now(),
            candidate=HARRIS,
            state=state,
        )
        miss = estimate_margin_error(weight_sum, target_full_error=0.038)
        print("State", state, "miss", miss)
        all_estimates[convert_state_name_to_state_code(state)] = miss
        t_scaling = calc_scale_factor_for_t_dist(
            5, miss
        )
        print("t scaling", t_scaling)
    avg_miss = sum(all_estimates.values()) / len(all_estimates)
    print("Average miss margin", avg_miss)
    print("Aberage miss share", avg_miss / 2)
    return sum(all_estimates.values()) / len(all_estimates)


if __name__ == "__main__":
    #print(find_weight_sum_for_day(pd.Timestamp.now(), candidate=HARRIS, state=None))
    #exit()
    #print(get_sim_custom_averages(pd.Timestamp.now()))
    #exit()
    #for state in swing_states:
    #    print(state)
    #    print(find_election_day_2020_weight_sum(state))
    #for weight in range(1, 25):
    #    print("weight", weight, "error", estimate_margin_error(weight, target_full_error=3.8))
    print(average_estimated_miss())
    exit()

    for state in swing_states:
        print(state)
        print(fit_linear_func_national_to_state(state))
    plot_poll_avgs()
    #print(average_election_day_2020_weight_sum())