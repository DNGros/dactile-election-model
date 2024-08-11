from typing_extensions import Counter
import itertools
from election_statics import HARRIS, REP, DEM
from simulate import simulate_election_mc, estimate_fracs


def main():
    sims = simulate_election_mc(
        dem_candidate=HARRIS,
        average_movement=0,
    )
    print("ELECTION TODAY")
    print_state_sets(sims)
    sims = simulate_election_mc(
        dem_candidate=HARRIS,
    )
    print("WITH MOVEMENT")
    print_state_sets(sims)


def print_state_sets(sims):
    frac, states = estimate_fracs(sims)
    print("Base frac dem:", frac)
    did_win_states = Counter()
    did_win_states_and_was_win = Counter()
    total_wins = 0
    for sim in sims:
        if sim.winner == DEM:
            total_wins += 1
        dem_win_states = [
            state
            for state, result in sim.election.state_results.items()
            if result.winner == DEM
        ]
        for i in range(1, len(dem_win_states) + 1):
            for combo in itertools.combinations(dem_win_states, i):
                did_win_states[combo] += 1
                if sim.winner == DEM:
                    did_win_states_and_was_win[combo] += 1

    print("Winning maps:")
    print("Unique sets", len(did_win_states))
    print("states, P(D_wins_states), P(D_win_states ∩ D_win_nationally), P(D_win_states | D_win_nationally), P(D_win_nationally | D_win_states)")
    for combo, count in did_win_states_and_was_win.most_common(10):
        win_count = did_win_states[combo]
        time_d_wins = win_count / len(sims)
        time_d_wins_and_wins = count / len(sims)
        frac = count / total_wins
        win_nationally_given_win_states = count / win_count
        combo_str = "{" + ", ".join(combo) + "}"
        print(f"{combo_str}, {time_d_wins * 100:.1f}%, {time_d_wins_and_wins*100:.1f}%, {frac*100:.1f}%, {win_nationally_given_win_states*100:.1f}%")
        #print(combo, round(frac*100,1), round(count / total_wins * 100, 1))


if __name__ == "__main__":
    main()