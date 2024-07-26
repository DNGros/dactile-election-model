from markupsafe import Markup

from election_statics import BIDEN, HARRIS
from harris.harris_explore import harris_national_table_html, get_avail_harris_biden_polls, \
    build_harris_national_table_df, get_average_harris_delta, harris_swing_state_table_html
from harris.typical_variances import ASSUMED_DAYS_TO_ELECTION, average_swing_all_cycle, find_max_swing_row, \
    get_state_on_election_day_2020_poll, get_2024_poll_num
from historical_elections import get_2020_election_struct
from hyperparams import default_movement_cur_cycle_average_multiple, harris_delta_error_default
from simulate import estimate_fracs, simulate_election_mc, PollMissKind
import pandas as pd


def get_article_vars():
    vals = {
        "cur_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "title": "Understanding a Harris vs Trump matchup. Modeling How The Race Would Begin.",
        "publish_date": "2024-07-19",
    }
    vals |= get_main_fracs()
    vals |= get_variance_vars()
    vals |= get_polls_vars()
    return vals


def get_variance_vars():
    max_swing_row = find_max_swing_row()
    election_2020 = get_2020_election_struct()
    assert max_swing_row['state'] == 'Wisconsin'
    return {
        "days_to_election": ASSUMED_DAYS_TO_ELECTION,
        "average_movement": round(average_swing_all_cycle(), 2),
        "max_swing_abs": round(max_swing_row['swing_abs'], 2),
        "max_swing_start": max_swing_row['start_date'].strftime("%B %d, %Y"),
        "max_swing_end": max_swing_row['end_date'].strftime("%B %d, %Y"),
        "wisconsin_election_day_2020": round(get_state_on_election_day_2020_poll("Wisconsin"), 1),
        "wisconsin_today": round(get_2024_poll_num("Wisconsin"), 1),
        "wisconsin_today_gap_from_2020": round(get_state_on_election_day_2020_poll("Wisconsin") - get_2024_poll_num("Wisconsin"), 1),
        "wisconsin_actual_margin": round(election_2020.state_results['WI'].get_dem_frac_of_major() * 100 - 50, 1),
        "default_movement_cur_cycle_average_multiple": default_movement_cur_cycle_average_multiple,
    }


def get_polls_vars():
    return {
        "num_harris_national_polls": len(build_harris_national_table_df()),
        "average_harris_delta": round(get_average_harris_delta(weighted=False)*100, 2),
        "average_harris_delta_weighted": round(get_average_harris_delta(weighted=True)*100, 2),
    }


def get_main_fracs():
    return {
        "biden_base_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=BIDEN,
            poll_miss=PollMissKind.POLL_MISS_TODAY_CORRELATED,
        ))[0] * 100),
        "biden_movement_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=BIDEN,
            poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        ))[0] * 100),
        "harris_point_estimate": round(estimate_fracs(simulate_election_mc(
            dem_candidate=HARRIS,
            poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        ))[0] * 100),
        'harris_delta_error_default': f"{harris_delta_error_default:.2f}",
    }


def make_all_harris_tables():
    return {
        "harris_national_table": Markup(harris_national_table_html()),
        "harris_swing_table": Markup(harris_swing_state_table_html()[0]),
    }

