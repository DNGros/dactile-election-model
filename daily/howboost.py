from election_statics import HARRIS, convert_state_name_to_state_code
from hyperparams import swing_states
from simulate import estimate_fracs, simulate_election_mc, DistributionParams
from whitmer.plotting import plot_election_map


def main():
    frac, states = estimate_fracs(simulate_election_mc(
        dem_candidate=HARRIS,
        state_dem_share_adjustment_dist={
            convert_state_name_to_state_code(state): DistributionParams(loc=0.0125, scale=0)
            for state in swing_states
        },
    ))
    plot_election_map(
        frac, states,
        title="Boost each state 1.25 share-points", candidate_name="Harris",
        show=True,
    )
    print(frac)


if __name__ == "__main__":
    main()