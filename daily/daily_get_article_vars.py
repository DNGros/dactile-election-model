import math

import numpy as np

from daily.custom_poll_avg import fit_linear_func_national_to_state, find_weight_sum_for_day, \
    find_election_day_2020_weight_sum, find_biden_dropout_day, average_estimated_miss
from election_statics import BIDEN, HARRIS, convert_state_name_to_state_code
from harris.harris_explore import build_harris_national_table_df, get_average_harris_delta
from harris.typical_variances import ASSUMED_DAYS_TO_ELECTION, average_swing_all_cycle, find_max_swing_row, \
    get_state_on_election_day_2020_poll, get_2024_poll_num, all_swing_sf, harris_cycle_moves
from historical_elections import get_2020_election_struct
from hyperparams import default_movement_cur_cycle_average_multiple, harris_delta_error_default, swing_states
from simulate import estimate_fracs, simulate_election_mc, PollMissKind, average_poll_miss
import pandas as pd

def get_article_vars():
    vals = {
        "cur_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "title": "What is the probability Harris wins? Building a Statistical Model.",
        "publish_date": "2024-07-26",
    }
    vals |= get_main_fracs()
    vals |= get_variance_vars()
    vals |= get_polls_vars()
    return vals


def get_variance_vars():
    max_swing_row_biden = find_max_swing_row()
    election_2020 = get_2020_election_struct()
    #assert max_swing_row['state'] == 'Wisconsin'
    max_state = max_swing_row_biden['state']
    swings_df = all_swing_sf()
    return {
        "days_to_election": ASSUMED_DAYS_TO_ELECTION,
        "average_movement": round(average_swing_all_cycle(HARRIS), 2),
        "average_movement_biden_2020": f"{swings_df[swings_df['cycle'] == 2020]['swing_abs'].mean():.2f}",
        "average_movement_biden_2024": round(swings_df[(swings_df['cycle'] == 2024) & (swings_df['candidate'] == BIDEN)]['swing_abs'].mean(), 2),
        "average_movement_harris_2024": round(harris_cycle_moves(), 2),
        "state_of_max_swing": max_state,
        "max_swing_abs": round(max_swing_row_biden['swing_abs'], 2),
        "max_swing_start": max_swing_row_biden['start_date'].strftime("%B %d, %Y"),
        "max_swing_end": max_swing_row_biden['end_date'].strftime("%B %d, %Y"),
        "max_state_election_day_2020": round(get_state_on_election_day_2020_poll(max_state), 1),
        "max_state_today": round(get_2024_poll_num(max_state), 1),
        "max_state_today_gap_from_2020": round(get_state_on_election_day_2020_poll(max_state) - get_2024_poll_num(max_state), 1),
        "max_state_actual_margin": round(election_2020.state_results[convert_state_name_to_state_code(max_state)].get_dem_frac_of_major() * 100 - 50, 1),
        "num_swing_states": len(swing_states),
        "actual_average_move": round(np.mean([
            average_swing_all_cycle(state=state, candidate=HARRIS)
            for state in swing_states
        ]), 2),
        "average_r2": str(round(np.mean([
            fit_linear_func_national_to_state(state)[2] ** 2
            for state in swing_states
        ]), 2)),
    }


def get_polls_vars():
    return {
        "num_national_polls_now": round(find_weight_sum_for_day(
            date=pd.Timestamp.now(),
            state=None,
            candidate=HARRIS,
        ), 1),
        "num_national_polls_elect2020": round(find_election_day_2020_weight_sum(), 1),
        'num_national_polls_dropout_day': round(find_biden_dropout_day(), 1),
        "average_expected_poll_miss": round(
            average_estimated_miss() / 2 * 100,  # Divide by 2 to go from margin to share
            1
        ),

    }


def get_main_fracs():
    return {
        "biden_movement_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=BIDEN,
            poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        ))[0] * 100),
        "harris_today_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=HARRIS,
            poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
            average_movement=0,
        ))[0] * 100),
        "measured_sim_poll_miss": round(average_poll_miss(
            PollMissKind.RECENT_CYCLE_CORRELATED,
        ) * 100, 1),
    }


if __name__ == "__main__":
    print(get_variance_vars())
