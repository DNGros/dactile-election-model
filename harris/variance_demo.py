import scipy
import matplotlib.pyplot as plt
import numpy as np

from get_poll_data import get_state_averages
from harris.typical_variances import get_2024_poll_num
from state_correlations import calc_scale_factor_for_t_dist


def make_variance_demo_plot(save_path = None):
    polls = {
        state: get_2024_poll_num(state) + 50
        for state in [
            "PA",
            "GA",
            "AZ",
            "NC",
            "FL",
            "WI",
        ]
    }
    polls = {
        state: poll
        for state, poll in polls.items()
        if poll < 50
    }
    # Sort by poll and take the first middle and last
    sorted_polls = sorted(polls.items(), key=lambda x: x[1], reverse=True)
    f_m_l = {}
    if len(sorted_polls) >= 4:
        # Actually take the 2nd
        f_m_l[sorted_polls[1][0]] = sorted_polls[1][1]
    else:
        # If there are only 3, take the 1st
        f_m_l[sorted_polls[0][0]] = sorted_polls[0][1]
    # Middle
    if len(sorted_polls) > 2:
        f_m_l[sorted_polls[len(sorted_polls) // 2][0]] = sorted_polls[len(sorted_polls) // 2][1]
    if len(sorted_polls) > 1:
        # Last
        f_m_l[sorted_polls[-1][0]] = sorted_polls[-1][1]
    # Reverse the order
    #polls = dict(reversed(list(f_m_l.items())))
    polls = f_m_l

    distributions = [
        scipy.stats.t(df=5, scale=calc_scale_factor_for_t_dist(5, 2)),
        scipy.stats.t(df=5, scale=calc_scale_factor_for_t_dist(5, 6.5)),
    ]

    fig, axes = plt.subplots(len(polls), 2, figsize=(5, (2.5/3)*len(polls)), sharex=False, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    x = np.linspace(30, 70, 1000)

    for i, (state, poll) in enumerate(polls.items()):
        for j, dist in enumerate(distributions):
            if len(polls) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            y = dist.pdf(x - poll)

            ax.plot(x, y, 'k-', lw=1.5)
            ax.fill_between(x[x < 50], y[x < 50], color='red', alpha=0.2)
            ax.fill_between(x[x >= 50], y[x >= 50], color='blue', alpha=0.2)

            ax.set_xlim(35, 65)
            ax.set_ylim(0, 0.2)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Only show x-axis ticks for bottom row
            if i != 2:
                ax.set_xticks([])
            else:
                ax.set_xticks([40, 50, 60])
                ax.set_xlabel("Dem Share of Vote")

            # Only show y-axis ticks for left column
            if j != 0:
                ax.set_yticks([])

            # Add state label inside the top left corner of the plot
            ax.text(0.05, 0.95, state, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    fontweight='normal')

            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, lw=1)

            if i == 0:
                if j == 0:
                    ax.set_title("Less Variance")
                else:
                    ax.set_title("More Variance")

            if i == 1 or len(polls) == 1:
                color = "black"
                if j == 0:
                    # Add a label
                    ax.text(0.72, 0.45, "Fewer outcomes\nabove 50", transform=ax.transAxes,
                            color=color, alpha=0.7, horizontalalignment='center',
                            fontsize=8)
                    # Add an arrow pointing to the distribution
                    ax.annotate("", xy=(53, 0.03), xytext=(55, 0.08),
                                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
                else:
                    # Add a label
                    ax.text(0.72, 0.45, "More outcomes\nabove 50", transform=ax.transAxes,
                            color=color, alpha=0.7, horizontalalignment='center',
                            fontsize=8)
                    # Add an arrow pointing to the distribution
                    ax.annotate("", xy=(53, 0.04), xytext=(55, 0.08),
                                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))

            # Calculate and annotate the mass on either side of 50
            #mass_below = dist.cdf(50 - poll)
            #mass_above = 1 - mass_below
            #ax.text(0.2, 0.5, f"{mass_below:.2f}", transform=ax.transAxes,
            #        color='red', alpha=0.7)
            #ax.text(0.8, 0.5, f"{mass_above:.2f}", transform=ax.transAxes,
            #        color='blue', alpha=0.7, horizontalalignment='right')

    # Remove overall title and adjust layout
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    make_variance_demo_plot()