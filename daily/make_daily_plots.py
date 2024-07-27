from pathlib import Path
import numpy as np

from matplotlib.ticker import FuncFormatter

from daily.custom_poll_avg import get_custom_polling_average
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
    plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'harris_assumed.html'),
        title=f"Harris Combined Estimate",
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
        title=f"If the Election was Today",
        candidate_name="Harris",
    )

    margin_df = get_margins_df_custom_avg(True)
    p = make_state_movements_plot(margin_df, plot_base / "state_movements_mobile.svg", is_mobile=True)
    p = make_state_movements_plot(margin_df, plot_base / "state_movements_desktop.svg", is_mobile=False)
    p = make_available_polls_plot(plot_base / "available_polls.svg")


def make_available_polls_plot(
    save_path=None,
):
    state = "National"
    cycle = 2024
    candidate = HARRIS
    df = get_margins_df_custom_avg(True)
    # Get unique states and cycles
    states = df['state'].unique()
    cycles = df['cycle'].unique()

    # Set up the plot
    num_states = len(states)
    fig, axes = plt.subplots(num_states, 2, figsize=(8, 2.5 * num_states), squeeze=False)
    fig.suptitle(
        'How Many Top-Quality Recent Polls\nDo We Have?',
        fontsize=17,
        fontweight='bold',
        # Put in the top left corner
        x=0.07,
        #y=0.98,
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
        sqrt_ax = axes[idx, 1]
        #ax2 = ax.twinx()  # Create a twin axis for sqrt values

        state_data = df[df['state'] == state]

        for cycle in cycles:
            cycle_data = state_data[state_data['cycle'] == cycle]
            for candidate in cycle_data['candidate'].unique():
                candidate_data = cycle_data[cycle_data['candidate'] == candidate]

                #color = 'blue' if candidate == BIDEN else 'purple'
                if candidate == BIDEN:
                    if cycle == 2024:
                        color = sns.color_palette()[0]
                    else:
                        color = sns.color_palette()[1]
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
                sqrt_ax.plot(
                    candidate_data['day_of_year'],
                    np.sqrt(candidate_data['total_weight_sum']),
                    color=color,
                    linestyle=linestyle,
                    label=label
                )

        # Set up the axes
        #ax.set_xlim(0, 366)
        #ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
        #ax.set_xticks(month_starts)
        #ax.set_xlabel('Month')
        ax.set_xticks(month_starts)
        ax.set_xticklabels(months)
        ax.set_ylabel('Weighted Count')
        sqrt_ax.set_ylabel('Sqrt of Weighted Count')
        if idx == 0:
            ax.set_title('Available Counts')
            sqrt_ax.set_title('~Amount of Information')
        #ax.set_title(f"{state}")
        # Make the title inside the plot
        ax.text(0.1, 0.9, state, horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes,
                fontsize=14, fontweight='bold')
        ax.set_ylim(0, df['total_weight_sum'].max() * 1.1)
        sqrt_ax.set_ylim(0, np.sqrt(df['total_weight_sum'].max() * 1.1))

        # Set up the sqrt axis
        #y_min, y_max = ax.get_ylim()
        #ax2.set_ylim(np.sqrt(y_min), np.sqrt(y_max))
        #ax2.set_ylabel('Sqrt of Total Weight Sum')


    # Add the legend
    fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        title='Legend'
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.975]
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    else:
        plt.show()


if __name__ == "__main__":
    make_available_polls_plot()

