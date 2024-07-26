# Harris be a plottin'
from pathlib import Path

from election_statics import BIDEN, HARRIS
from harris.compare_moves import plot_average_move_to_win_prob, plot_average_harris_delta_error
from harris.typical_variances import make_state_movements_plot
from harris.variance_demo import make_variance_demo_plot
from hyperparams import harris_delta_error_default
from simulate import estimate_fracs, simulate_election_mc, PollMissKind
from whitmer.plotting import plot_election_map

cur_path = Path(__file__).parent.absolute()
map_base = cur_path / 'harrisarticlegen/imgs/maps'
map_base.mkdir(exist_ok=True, parents=True)
plot_base = cur_path / 'harrisarticlegen/imgs/plots'


def make_all_harris_plots():
    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss=PollMissKind.POLL_MISS_TODAY_CORRELATED,
    ))
    show = False
    p = plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'biden_base.html'),
        show=show,
        title="If the election was today (polling avg on July 19th)\nusing state-correlated errors:",
        candidate_name="Biden",
    )

    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
    ))
    p = plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'biden_assumed_move.html'),
        title="Assuming an 2x-average move",
        candidate_name="Biden",
    )

    # variance plot
    p = make_state_movements_plot(plot_base / "state_movements_mobile.svg", is_mobile=True)
    p = make_state_movements_plot(plot_base / "state_movements_desktop.svg", is_mobile=False)
    # Moves
    p = plot_average_move_to_win_prob(
        save_path=plot_base / "average_move_to_win_prob.svg",
    )
    p = make_variance_demo_plot(plot_base / "variance_demo.svg")
    p = plot_average_harris_delta_error(plot_base / "harris_delta_error.svg")

    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=HARRIS,
        poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        correlated_chaos_avg_change=harris_delta_error_default,
    ))
    p = plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'harris_assumed.html'),
        title=f"Harris Estimate ({harris_delta_error_default})",
        candidate_name="Harris",
    )


if __name__ == "__main__":
    make_all_harris_plots()