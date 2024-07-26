from election_statics import BIDEN, WHITMER
from simulate import estimate_fracs, simulate_election_mc, PollMissKind
import pandas as pd


def get_article_vars():
    vals = {
        "cur_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    vals |= get_main_fracs()
    return vals


def get_main_fracs():
    return {
        "biden_base_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=BIDEN,
            poll_miss=PollMissKind.SIMPLE_MISS,
        ))[0] * 100),
        "biden_adjusted_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=BIDEN,
            poll_miss=PollMissKind.ADJUSTED,
        ))[0] * 100),
        "whitmer_adjusted_prob": round(estimate_fracs(simulate_election_mc(
            dem_candidate=WHITMER,
            poll_miss=PollMissKind.ADJUSTED,
        ))[0] * 100),
    }
