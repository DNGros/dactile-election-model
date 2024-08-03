from pathlib import Path
import math
import pandas as pd
import numpy as np

from matplotlib.ticker import FuncFormatter

from daily.custom_poll_avg import get_custom_polling_average, find_biden_dropout_day, find_weight_sum_for_day, \
    find_election_day_2020_weight_sum
from election_statics import HARRIS, BIDEN
from harris.typical_variances import make_state_movements_plot, get_margins_df_custom_avg, election_day
from simulate import estimate_fracs, simulate_election_mc, PollMissKind
from whitmer.plotting import plot_election_map
import seaborn as sns
import matplotlib.pyplot as plt


cur_path = Path(__file__).parent.absolute()
map_base = cur_path / 'dailyarticlegen/imgs/maps'
map_base.mkdir(exist_ok=True, parents=True)
plot_base = cur_path / 'dailyarticlegen/imgs/plots'


def make_all_daily_plots():
    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=HARRIS,
        poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
    ))
    today = pd.Timestamp.now()
    plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'harris_assumed.html'),
        title=f"Harris Combined Estimate ({(today.strftime('%b %d'))})",
        candidate_name="Harris",
    )

    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=HARRIS,
        poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        average_movement=0,
    ))
    plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'harris_today.html'),
        title=f"If the Election was Today ({(today.strftime('%b %d'))})",
        candidate_name="Harris",
    )

    margin_df = get_margins_df_custom_avg(True)
    p = make_state_movements_plot(margin_df, plot_base / "state_movements_mobile.svg", is_mobile=True)
    p = make_state_movements_plot(margin_df, plot_base / "state_movements_desktop.svg", is_mobile=False)
    p = make_available_polls_plot(plot_base / "available_polls.svg")


def make_available_polls_plot(save_path=None):
    df = get_margins_df_custom_avg(True)
    #eational_avg = get_margins_df_custom_avg(False)
    #df = pd.concat([national_avg, df])
    states = list(df['state'].unique())
    cycles = df['cycle'].unique()

    num_states = len(states)
    fig, axes = plt.subplots(
        num_states, 2, figsize=(7, 2.25 * num_states), squeeze=False,
        gridspec_kw={'width_ratios': [3, 1]}
    )
    fig.suptitle(
        'How Many Top-Quality Recent Polls\nDo We Have?',
        fontsize=17,
        fontweight='bold',
        x=0.07,
        y=0.967,
        horizontalalignment='left',
    )
    fig.text(
        s='Figure 1:',
        fontsize=17,
        fontweight='normal',
        x=0.07,
        y=0.973,
        horizontalalignment='left',
    )

    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', '']
    start_month = 4
    months = months[start_month:]
    month_starts = month_starts[start_month:]
    month_starts.append(election_day[2024])

    df = df[df['day_of_year'] >= month_starts[0]]
    legend_handles = []

    for idx, state in enumerate(states):
        ax = axes[idx, 0]
        bar_ax = axes[idx, 1]

        state_data = df[df['state'] == state]

        for cycle in cycles:
            cycle_data = state_data[state_data['cycle'] == cycle]
            for candidate in cycle_data['candidate'].unique():
                candidate_data = cycle_data[cycle_data['candidate'] == candidate]

                if candidate == BIDEN:
                    color = sns.color_palette()[0] if cycle == 2024 else sns.color_palette()[1]
                elif candidate == HARRIS:
                    color = sns.color_palette()[6]
                linestyle = '-' if cycle == 2024 else '--'

                label = f"{candidate} {cycle}"
                line, = ax.plot(
                    candidate_data['day_of_year'],
                    candidate_data['total_weight_sum'],
                    color=color,
                    linestyle=linestyle,
                    label=label
                )
                if idx == 0:
                    legend_handles.append(line)

        # Set up the line plot
        ax.set_xticks(month_starts)
        ax.set_xticklabels(months)
        ax.set_ylabel('Weighted Count')
        if idx == 0:
            ax.set_title('Available Counts')
        ax.text(0.1, 0.9, state, horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes,
                fontsize=14, fontweight='bold')

        # Calculate the maximum value
        max_value = df['total_weight_sum'].max() * 1.1
        # Round up to the nearest 5
        rounded_value = math.ceil(max_value / 5) * 5
        # Set the y-axis limit
        ax.set_ylim(0, rounded_value)

        # Set up the bar plot for key values
        key_values = {
            'Biden\nDropout': find_biden_dropout_day(state),
            'Harris\nNow': find_weight_sum_for_day(
                date=pd.Timestamp.now(),
                state=state,
                candidate=HARRIS,
            ),
            '2020\nElection\nDay': find_election_day_2020_weight_sum(state)
        }

        bar_colors = [sns.color_palette()[0], sns.color_palette()[6], sns.color_palette()[1]]  # You can adjust these colors
        bars = bar_ax.bar(key_values.keys(), np.sqrt(list(key_values.values())), color=bar_colors)
        bar_ax.set_ylabel('Sqrt Weighted Count')
        if idx == 0:
            bar_ax.set_title('Est. Available\nInformation')
        bar_ax.set_ylim(0, np.sqrt(df['total_weight_sum'].max() * 1.1) * 1.1)

        # Rotate x-axis labels for better readability
        bar_ax.set_xticklabels(
            key_values.keys(),
            # smaller font
            fontsize=7,
        )

        # Add value labels on top of each bar
        #for bar in bars:
        #    height = bar.get_height()
        #    bar_ax.text(bar.get_x() + bar.get_width() / 2., height,
        #                f'{height:.2f}',
        #                ha='center', va='bottom')

    # Add the legend
    fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        title='Legend'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    else:
        plt.show()


if __name__ == "__main__":
    #make_all_daily_plots()
    make_available_polls_plot()

