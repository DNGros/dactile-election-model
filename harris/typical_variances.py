import functools
import math
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import numpy as np
from fontTools.misc.plistlib import end_date
from graphviz._compat import Literal

from daily.custom_poll_avg import get_custom_polling_average
from election_statics import convert_state_name_to_state_code, BIDEN, HARRIS
import pandas as pd
import seaborn as sns

from harris.margins import get_margins_df
from hyperparams import harris_article_calc_date, swing_states, dropout_day, harris_start_display_date
from util import get_only_value
from joblib import Memory
from pathlib import Path

cur_path = Path(__file__).parent.absolute()

cache = Memory(cur_path / "../cache")

#swing_states += ['Texas']
election_day = {
    2024: 309,
}
ASSUMED_DAYS_TO_ELECTION = election_day[2024] - pd.Timestamp.now().day_of_year


def calc_swing_df(
    df,
    day_lag=ASSUMED_DAYS_TO_ELECTION
):
    # Print the results
    #print(f"Maximum {day_lag}-day swings:")
    all_averages = []
    all_swings_df = []
    for cycle in df['cycle'].unique():
        for state in swing_states:
            group = df[(df['state'] == state) & (df['cycle'] == cycle)]
            if len(group) == 0:
                continue
            swings_df = calculate_swing(group, day_lag)
            swings_df['cycle'] = cycle
            swings_df['state'] = state
            swings_df['swing_abs'] = np.abs(swings_df['swing'])
            swings_df_filt = swings_df[swings_df['has_full_days']]
            if len(swings_df_filt) == 0:
                continue
            swings = np.abs(swings_df_filt['swing'])
            average_swing = np.mean(swings)
            all_averages.append(average_swing)
            all_swings_df.append(swings_df)
    #print(f"Total Average swing: {np.mean(all_averages):.2f}")
    return pd.concat(all_swings_df)


@functools.cache
def average_swing_all_cycle(
    candidate,
    state=None,
):
    """Tries to estimate the expected average move in vote share
    for a given candidate in a given state"""
    assert candidate in (BIDEN, HARRIS)
    swings_df = all_swing_df()  # Will just contain BIDEN's cycles
    assert state is None or len(state) > 2, "Expected state name not code"
    if state:
        swings_df = swings_df[swings_df['state'] == state]
    # Filter out early days where not actually full days
    swings_df = swings_df[swings_df['has_full_days']]
    # Average each cycle
    mean_swing = swings_df.groupby('cycle')['swing_abs'].mean().reset_index()
    if candidate == BIDEN:
        this_estimate = mean_swing['swing_abs'].mean()
    else:
        # Estimate from the Harris 2024 data
        harris_estimate = harris_cycle_moves(state)
        # Combine biden and harris, more weight on Harris
        this_estimate = (mean_swing['swing_abs'].mean() * 2 + harris_estimate) / 3
    if state is not None:
        # We don't want to overfit to a state. Combine it also with the national estimate
        average_swing_national = average_swing_all_cycle(candidate, state=None)
        this_estimate = (this_estimate + average_swing_national) / 2
    return this_estimate


@functools.cache
def all_swing_df():
    """A dataframe with the amount of days-to-election moves"""
    df = get_margins_df_custom_avg(True)
    df = df[df['candidate'] == BIDEN]
    swings_df = calc_swing_df(df).reset_index()
    swings_df['candidate'] = BIDEN
    return swings_df


@functools.cache
def find_max_swing_row(state: str = None, cycle = None) -> dict | None:
    swings_df = all_swing_df()
    if state:
        swings_df = swings_df[swings_df['state'] == state]
    if cycle:
        swings_df = swings_df[swings_df['cycle'] == cycle]
    swings_df = swings_df[swings_df['has_full_days']]
    if len(swings_df) == 0:
        return None
    max_indices = swings_df['swing_abs'].idxmax()
    max_swing_row = swings_df.loc[max_indices]
    return max_swing_row


def get_state_on_election_day_2020_poll(state: str):
    df = get_margins_df_custom_avg(True)
    df = df[(df['state'] == state) & (df['cycle'] == 2020)]
    max_day = df['day_of_year'].max()
    df = df[df['day_of_year'] == max_day]
    return get_only_value(df['d_r_share_margin'].values)


def get_2024_poll_num(state: str):
    df = get_margins_df_custom_avg(True)
    df = df[
        (df['state_code'] == convert_state_name_to_state_code(state))
        & (df['cycle'] == 2024)
        & (df['candidate'] == HARRIS)
    ]
    max_day = df['day_of_year'].max()
    df = df[df['day_of_year'] == max_day]
    return get_only_value(df['d_r_share_margin'].values)


def calculate_swing(group, days):
    group = group.sort_values('day_of_year')
    vals = []
    for i in range(len(group)):
        current_day = group.iloc[i]['day_of_year']
        current_value = group.iloc[i]['d_r_share_margin']

        # Find the closest day within the past 'days' days
        past_days = group[
            (group['day_of_year'] < current_day) &
            (group['day_of_year'] >= current_day - days)
        ]

        if not past_days.empty:
            closest_past_day = past_days.iloc[0]
            swing = current_value - closest_past_day['d_r_share_margin']
            vals.append({
                'swing': swing,
                'start_day': closest_past_day['day_of_year'],
                'end_day': current_day,
                'start_date': closest_past_day['date'],
                'end_date': group.iloc[i]['date'],
                'has_full_days': current_day - closest_past_day['day_of_year'] >= days - 1,
                'start_value': closest_past_day['d_r_share_margin'],
                'end_value': current_value,
            })

    return pd.DataFrame(vals)


@functools.cache
@cache.cache()
def get_margins_df_custom_avg(
    only_swing_states: bool = True,
):
    """Same as get_margins but using our custom avg"""
    if only_swing_states:
        avgs = [
            *[
                get_custom_polling_average(
                    candidate=candidate,
                    cycle=cycle,
                    state=state,
                    include_national=True,
                )
                for state in swing_states
                for cycle, candidate in ((2020, BIDEN), (2024, BIDEN), (2024, HARRIS))
            ]
        ]
    else:
        avgs = [
            *[
                get_custom_polling_average(
                    candidate=candidate,
                    cycle=cycle,
                    state=None,
                    include_national=True,
                )
                for cycle, candidate in ((2020, BIDEN), (2024, BIDEN), (2024, HARRIS))
            ]
        ]
    df = pd.concat(avgs)
    df['day_of_year'] = df['date'].dt.dayofyear
    # Filter to March 1 since that's when the 2024 data starts
    df = df[df['day_of_year'] >= 60].copy()
    # Get the total pct for each (date, state)
    df['percent_of_d_r_share'] = df['average']
    df['d_r_share_margin'] = df['percent_of_d_r_share'] - 50
    if only_swing_states:
        df['state_code'] = df['state'].apply(convert_state_name_to_state_code)
    else:
        df['state_code'] = None
    # Dropout times
    df = df[(df['candidate'] != BIDEN) | (df['date'] < dropout_day)]
    df = df[(df['candidate'] != HARRIS) | (df['date'] >= harris_start_display_date)]
    return df


def make_state_movements_plot(
    margins_df,
    save_path=None,
    is_mobile=False,
):
    df = margins_df

    cycles = df['cycle'].unique()
    num_states = len(swing_states)
    num_cycles = len(cycles)

    # Define the base size for each subplot
    base_width = 6
    base_height = 3

    if is_mobile:
        base_width /= 1.5
        base_height /= 1.5

    if is_mobile:
        num_rows = num_states
        num_cols = num_cycles
        figsize = (base_width * num_cycles, base_height * num_states)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    else:
        num_rows = math.ceil(num_states / 2)
        num_cols = num_cycles * 2
        figsize = (base_width * num_cols, base_height * num_rows)
        # Define the GridSpec layout with an extra column for spacing
        gs = gridspec.GridSpec(
            num_rows, num_cols + 1, width_ratios=[1, 1, 0.1, 1, 1],
            hspace=0.4, wspace=0.125
        )
        fig = plt.figure(figsize=figsize)

        axes = []
        for i in range(num_rows):
            row_axes = []
            for j in range(num_cols):
                col_idx = j + 1 if j > 1 else j  # Shift index for columns after the first
                row_axes.append(fig.add_subplot(gs[i, col_idx]))
            axes.append(row_axes)
        axes = np.array(axes)


    # Ensure axes is always a 2D array
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(num_rows, num_cols)

    state_cycle_to_ax = {}
    for i, state in enumerate(swing_states):
        for j, cycle in enumerate(cycles):
            if is_mobile:
                ax = axes[i, j]
            else:
                row = i // 2
                ax = axes[row, j + num_cycles * (i % 2)]
            state_cycle_to_ax[(state, cycle)] = ax
            for _, candidate in enumerate(df.candidate.unique()):
                state_cycle_data = df[
                    (df['state'] == state)
                    & (df['cycle'] == cycle)
                    & (df['candidate'] == candidate)
                ]
                if state_cycle_data.empty:
                    continue
                sns.lineplot(
                    data=state_cycle_data,
                    x='day_of_year',
                    y='percent_of_d_r_share',
                    ax=ax,
                    linewidth=3,
                    color='blue' if candidate == BIDEN else 'purple',
                )

            if cycle == 2024:
                # Add a dashed line on dropout day
                ax.axvline(dropout_day.day_of_year, color='black', linestyle='--')

            #ax.set_title(f"{state} - {cycle}")
            # Put the title in the top left corner inside the plot
            ax.text(
                0.02, 0.98,
                f"{state} | {cycle}",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=12,
                fontweight='bold',
            )
            #ax.set_xlabel("Day of Year")
            ax.set_xlabel("")
            if j % 2 == 0:
                ax.set_ylabel("Dem Share")
            else:
                ax.set_ylabel("")
            ax.set_ylim(50-7, 50+7)
            ax.set_xlim(31+15, election_day[2024])
            # Make y-ticks every 2
            ax.set_yticks(np.arange(50-6, 50+7, 2))

    # show a tick each month
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', '']
    start_month = 2
    months = months[start_month:]
    month_starts = month_starts[start_month:]
    month_starts.append(election_day[2024])
    for ax in axes.flatten():
        ax.set_xticks(month_starts)
        ax.set_xticklabels(months)
    # Annotate a line at middle
    for ax in axes.flatten():
        ax.axhline(50, color='black', linewidth=1)

    swings_df = calc_swing_df(df)
    swings_df = swings_df[swings_df['has_full_days']]
    mean_max_swing = swings_df.groupby(['cycle', 'state'])['swing_abs'].agg(['mean', 'max'])
    # Annotate a span for the largest swing in Wisconsin
    max_swing_row = find_max_swing_row()
    # Add bracket for the largest swing
    true_max_state = max_swing_row['state']
    true_max_cycle = max_swing_row['cycle']

    # Add span brackets
    for (state, cycle), ax in state_cycle_to_ax.items():
        # Annotate a span for the largest swing in Wisconsin
        max_swing_row = find_max_swing_row(state, cycle)
        if max_swing_row is None:
            continue
        max_start_day = max_swing_row['start_day']
        max_end_day = max_swing_row['end_day']
        max_swing = max_swing_row['swing']
        lowest_val = min(max_swing_row['start_value'], max_swing_row['end_value']) + 50
        highest_val = max(max_swing_row['start_value'], max_swing_row['end_value']) + 50
        y_min = min(ax.get_ylim())
        y_max = max(ax.get_ylim())
        y_range = y_max - y_min

        # Calculate positions for the bracket
        bracket_height = y_range * 0.05
        bracket_bottom = lowest_val - y_range * 0.25
        if lowest_val > y_min + y_range * 0.3:
            bracket_vertical = bracket_bottom + bracket_height
        else:
            bracket_bottom = highest_val + y_range * 0.1
            bracket_vertical = bracket_bottom - bracket_height
        print(f"State: {state}, cycle: {cycle}, max_swing: {max_swing}")
        if state == true_max_state and cycle == true_max_cycle:
            color = 'red'
            lw = 3
            fontweight = 'bold'
        else:
            color = 'grey'
            lw = 2
            fontweight = 'normal'


        # Draw the bracket
        ax.plot([max_start_day, max_start_day], [bracket_bottom, bracket_vertical],
                color=color, lw=lw)
        ax.plot([max_end_day, max_end_day], [bracket_bottom, bracket_vertical],
                color=color, lw=lw)
        ax.plot([max_start_day, max_end_day],
                [bracket_bottom, bracket_bottom], color=color, lw=lw)

        # Add text label
        mid_point = (max_start_day + max_end_day) / 2
        ax.text(mid_point, bracket_bottom + y_range * 0.05, f'Largest move: {max_swing:.2f}',
                horizontalalignment='center', color=color, fontweight=fontweight)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            save_path,
            transparent=True,
            bbox_inches='tight',
        )
        plt.close()
        return save_path
    else:
        plt.show()


def harris_cycle_moves(state=None) -> float:
    """Given the harris data so far, try to estimate a mean absolute move.
    This involves a process of random walks using bootstrapped samples
    of movement so far. This process is pretty rough. There's probably
    a more principled way to model this, but for now this version will work.
    """
    df = get_margins_df_custom_avg()
    df = df[df.candidate == HARRIS]
    df = df[df.cycle == 2024]
    if state is not None:
        df = df[df.state == state]
    today = pd.Timestamp.now()
    days_to_election = election_day[2024] - today.day_of_year
    days_since_dropout = today.day_of_year - dropout_day.day_of_year
    dropout_to_election = election_day[2024] - dropout_day.day_of_year
    # Get moves on a subset of the days we will use for random walks
    day_lag = int(days_since_dropout / 4)
    swing_df = calc_swing_df(df, day_lag)
    swing_df = swing_df[swing_df['has_full_days']]
    # Get a numpy of swings
    swings = np.array(swing_df['swing'].values)
    # Sample bootstrapped random walks. We sample day spans from our
    # data so far
    num_samples = 5000
    num_steps = days_to_election / day_lag
    # We are going to assume some trend following by making two steps
    # in the direction of a single sample
    trend_following_steps = 2
    actual_steps_sampled = math.ceil(num_steps / trend_following_steps)
    np.random.seed(len(str(state)))
    samples = np.random.choice(
        swings, (num_samples, actual_steps_sampled),
        replace=True,
    )
    # Calculate the average of each sample
    sample_sums = np.sum(samples * trend_following_steps, axis=1)
    # Calculate the average of the sample means
    bootstrap_walks = np.mean(np.abs(sample_sums))
    print("Bootstrap walks mean", bootstrap_walks)
    # If we have enough days we can directly average the move
    can_direct_estimate = days_to_election + 1 < days_since_dropout
    if can_direct_estimate:
        direct_swing_df = calc_swing_df(df, days_to_election)
        direct = direct_swing_df['swing_abs'].mean()
        direct_weight = (days_since_dropout / 2) / (dropout_to_election / 2)
        direct_weight *= 0.75
        direct_weight = max(0, min(1, direct_weight))
        return direct * direct_weight + bootstrap_walks * (1 - direct_weight)
    return bootstrap_walks


if __name__ == "__main__":
    v = {
        state: average_swing_all_cycle(HARRIS, state)
        for state in swing_states
    }
    print(v)
    print("average", np.mean(list(v.values())))
    print("harris national average", average_swing_all_cycle(HARRIS))
    exit()
    #print(get_margins_df_custom_avg())
    print(harris_cycle_moves())
    print("average swing all cycles", average_swing_all_cycle(HARRIS))
    #print(get_state_on_election_day_2020_poll("Wisconsin"))
    #make_state_movements_plot()