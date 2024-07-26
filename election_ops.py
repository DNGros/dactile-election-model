import dataclasses

from election_statics import get_electoral_votes, get_all_state_codes
from election_structs import Election
from historical_elections import get_2020_election_struct


@dataclasses.dataclass
class ElectionScore:
    winner: str
    dem_electoral_votes: int
    rep_electoral_votes: int
    state_to_winner: dict[str, str] = None
    election: Election = None


    @classmethod
    def from_election(
        cls,
        election: Election,
        baseline_election: Election = None,
    ) -> 'ElectionScore':
        dem_electoral_votes = 0
        rep_electoral_votes = 0
        _, state_to_electors = get_electoral_votes()
        state_to_winner = {}
        for state_code in get_all_state_codes():
            if state_code in election.state_results:
                state_result = election.state_results[state_code]
                winner = state_result.winner
            elif baseline_election and state_code in baseline_election.state_results:
                state_result = baseline_election.state_results[state_code]
                winner = state_result.winner
            else:
                raise ValueError(f"State {state_code} not found in election or baseline election")
            if winner == "DEM":
                dem_electoral_votes += state_to_electors[state_code]
            elif winner == "REP":
                rep_electoral_votes += state_to_electors[state_code]
            state_to_winner[state_code] = winner
        return cls(
            winner="DEM" if dem_electoral_votes > rep_electoral_votes else "REP",
            dem_electoral_votes=dem_electoral_votes,
            rep_electoral_votes=rep_electoral_votes,
            election=election,
            state_to_winner=state_to_winner,
        )


if __name__ == "__main__":
    elect2020 = get_2020_election_struct()
    print(ElectionScore.from_election(elect2020))
