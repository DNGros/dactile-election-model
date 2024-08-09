"""
Calculate the flipped states numbers similiar to Silver's scenario table
"""
from collections import Counter
from election_statics import HARRIS
from election_structs import Election, StateResult
from historical_elections import get_2020_election_struct
from simulate import simulate_election_mc


def main():
    sims = simulate_election_mc(
        dem_candidate=HARRIS,
    )
    elect2020 = get_2020_election_struct()
    has_any_flip_count = 0
    has_any_flipped_to_dem_count = 0
    has_any_flipped_to_rep_count = 0
    state_dem_flip_count = Counter()
    state_dem_closest_flip_count = Counter()
    for sim in sims:
        flipped_results = get_flipped_states(sim.election, elect2020)
        flipped_results.sort(key=lambda result: abs(result.frac_rep - result.frac_dem))
        flipped_to_dem = [result for result in flipped_results if result.winner == "DEM"]
        flipped_to_rep = [result for result in flipped_results if result.winner == "REP"]
        if flipped_results:
            has_any_flip_count += 1
        if flipped_to_dem:
            has_any_flipped_to_dem_count += 1
            state_dem_flip_count.update([r.state for r in flipped_to_dem])
            state_dem_closest_flip_count[flipped_to_dem[0].state] += 1
        if flipped_to_rep:
            has_any_flipped_to_rep_count += 1
    flip_to_dem_frac = has_any_flipped_to_dem_count / len(sims)
    print(f"Has any flipped to dem: {flip_to_dem_frac*100:.2f}%")
    print(f"Has any flipped to rep: {has_any_flipped_to_rep_count / len(sims)*100:.2f}%")
    print(f"Has any flipped state: {has_any_flip_count / len(sims) * 100:.2f}%")
    print("How likely is each state to be the narrowest flip to Dem")
    for state, count in state_dem_closest_flip_count.most_common(10):
        state_frac = count / len(sims)
        print(f"{state}: {state_frac * 100:.2f}% ({state_frac / flip_to_dem_frac * 100:.2f}% of cases with dem flip)")
    print("How often does each state flip to Dem (don't care about narrowness)")
    for state, count in state_dem_flip_count.most_common(10):
        state_frac = count / len(sims)
        print(f"{state}: {state_frac * 100:.2f}% ({state_frac / flip_to_dem_frac * 100:.2f}% of cases with dem flip)")


def get_flipped_states(
    new_election: Election,
    baseline_election: Election,
) -> list[StateResult]:
    flipped_results = []
    for state_code, result in new_election.state_results.items():
        baseline_result = baseline_election.state_results[state_code]
        if result.winner != baseline_result.winner:
            flipped_results.append(result)
    return flipped_results


if __name__ == "__main__":
    main()