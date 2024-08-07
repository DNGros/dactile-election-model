"""
Try to estimate how much a potential VP boost could matter
"""
import concurrent
from pprint import pprint
import numpy as np
from tqdm import tqdm
from election_ops import ElectionScore
from election_statics import HARRIS, DEM, REP, state_code_to_state_name, convert_state_name_to_state_code
from election_structs import StateResult, Election
from historical_elections import get_2020_election_struct
from hyperparams import swing_states
from simulate import PollMissKind, estimate_fracs, simulate_election_mc, DistributionParams
from state_correlations import get_correlation_matrix_pow_for_one_state, corr2cov
from whitmer.plotting import interpolate_color
from scipy import stats

from joblib import Memory
from pathlib import Path
cur_path = Path(__file__).parent.absolute()

cache = Memory(cur_path / "../cache")

default_params = {
    "dem_candidate": HARRIS,
    "poll_miss": PollMissKind.RECENT_CYCLE_CORRELATED,
    "average_movement": 0,
}


def compute_win_chance_shifted(mean, std, state, sims, corr_dampening=None):
    """Alters the simulation values with some boost in a given state.
    Follows a normal distribution with a given mean and std.

    We also can optionally also use a correlation dampening factor. Given the
    correlation dampening factor we raise the correlations by this power, and divide
    the magnitiude of shifts by this value.

    Note this is different from how we did it with Whitmer (and probably the more right way).
    We are actually sampling from a multivariable distribution here rather than
    just scaling the shift by the correlation factor. This better allows for cases
    where the VP helps in one state but hurts in another.
    """
    params = default_params.copy()

    # Rerunning method
    #params['state_dem_share_adjustment_dist'] = {
    #    state: DistributionParams(mean, std)
    #}
    #win_chance, states = estimate_fracs(simulate_election_mc(**params))

    # Just do it with adjusting states in old sims
    baseline_election = get_2020_election_struct()
    new_sims = []
    swing_state_codes = [convert_state_name_to_state_code(state) for state in swing_states]
    if corr_dampening is None:
        dist = stats.norm(loc=mean, scale=std)
        print("mean", mean, "std", std)
        shifts = dist.rvs(size=len(sims))
    else:
        # In this version we assume some correlated shifts scaled to some power
        dist_and_states = []
        # Get the correlation distribution using each source
        for src in ['538', 'eco']:
            cor_mat, states = get_correlation_matrix_pow_for_one_state(
                src, state, corr_dampening, states=swing_state_codes
            )
            dist = stats.multivariate_normal(
                cov=corr2cov(cor_mat, 1),
                mean=[mean]*len(states)
            )
            dist_and_states.append((dist, states))
        assert dist_and_states[0][1] == dist_and_states[1][1]
        assert len(sims) % 2 == 0, f"Need an even number of sims {len(sims)}"
        half_shifts = dist_and_states[0][0].rvs(size=int(len(sims) / 2))
        other_half_shifts = dist_and_states[1][0].rvs(size=int(len(sims) / 2))
        shifts = np.concatenate([half_shifts, other_half_shifts])
        assert shifts.shape == (len(sims), len(swing_state_codes))


    for i, sim in enumerate(tqdm(sims, desc="Adjusting state", total=len(sims))):
        #if mean == 0 and std == 0:
        #    new_sims.append(sim)
        #    continue
        election = sim.election
        results = election.state_results.copy()
        for state_ind, change_state in enumerate(swing_state_codes):
            if corr_dampening is None and change_state != state:
                continue  # Without correlations we only shift the given state
            old_state_val = election.state_results[change_state]
            old_dem_share = old_state_val.frac_dem
            if corr_dampening is None:
                new_dem_share = old_dem_share + shifts[i]
            else:
                this_shift = shifts[i][state_ind] * std + mean
                if state != change_state:
                    this_shift /= corr_dampening
                new_dem_share = old_dem_share + this_shift
            new_dem_share = min(1, max(0, new_dem_share))
            results[change_state] = StateResult(
                state=change_state,
                winner=DEM if new_dem_share > 0.5 else REP,
                frac_dem=new_dem_share,
                frac_rep=1 - new_dem_share,
                frac_other=0,
                total_votes=old_state_val.total_votes,
                from_beliefs=old_state_val.from_beliefs,
            )
        new_election = Election(
            state_results=results,
            remaining_states=election.remaining_states,
            dem_candidate=election.dem_candidate
        )
        new_sims.append(ElectionScore.from_election(new_election, baseline_election))
    win_chance, states = estimate_fracs(new_sims)
    return (mean, std, win_chance, states[state])


def get_particular_value_state_shift(
    state,
    mean,
    std,
    corr_dampening=None,
):
    #sims = simulate_election_mc(
    #    dem_candidate=HARRIS,
    #    poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
    #    average_movement=0,
    #)
    sims = simulate_election_mc(**default_params)
    _, _, national, state = compute_win_chance_shifted(mean, std, state, sims, corr_dampening)
    return national, state



means = [0, 0.0025, 0.005, 0.01, 0.015, 0.05]
#means = [0, 0.0025, 0.005, 0.01, 0.1]
std_devs = [0, 0.01, 0.02, 0.04]

#means = [0]
#std_devs = [0]

@cache.cache()
def vp_adjust_vals_dict(state, corr_dampening=None):
    print("get base sims")
    sims = simulate_election_mc(**default_params)
    print("done")
    #pprint({
    #    (mean, std): compute_win_chance(mean, std, state, sims, corr_factor)
    #    for std in std_devs
    #    for mean in means
    #})
    #exit()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            (mean, std): executor.submit(compute_win_chance_shifted, mean, std, state, sims, corr_dampening)
            for std in std_devs
            for mean in means
        }
        win_chances = {
            (mean, std): (future.result()[2], future.result()[3])
            for (mean, std), future in futures.items()
        }
    return win_chances


def make_vp_shift_table(state, corr_dampening=None):
    win_chances = vp_adjust_vals_dict(state, corr_dampening)
    lines = ["<table>"]
    lines.append("<tr>")
    lines.append("<th>Mean Adjustment →<br>σ ↓</th>")
    for mean in means:
        lines.append(f"<th>{mean*100}</th>")
    lines.append("</tr>")
    state_emoji = {
        "PA": "🪨",
        "AZ": "🌵",
    }[state]

    for std in std_devs:
        lines.append("<tr>")
        lines.append(f"<td>{std*100}</td>")
        for mean in means:
            win_chance_nat, win_chance_state = win_chances[(mean, std)]
            color = interpolate_color(fraction=win_chance_nat, alpha=0.2)
            lines.append(f"<td style='background-color:{color}'>"
                         f"{round(win_chance_nat * 100)}%🇺🇸<br>{round(win_chance_state * 100)}%{state_emoji}</td>")
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


if __name__ == "__main__":
    print("no cor")
    vals = vp_adjust_vals_dict("PA", corr_dampening=None)
    pprint({
        (mean, std): (round(val[0]*100,1), round(val[1]*100,1))
        for (mean, std), val in vals.items()
        if std == 0
    })
    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=HARRIS,
        poll_miss=PollMissKind.RECENT_CYCLE_CORRELATED,
        average_movement=0,
    ))
    print(overall_frac, state_frac)
    exit()
    #print("with cor")
    #print(vp_adjust_vals_dict("PA", corr_dampening=3))
    #print(make_vp_shift_table("PA"))
    #print(make_vp_shift_table("AZ"))
