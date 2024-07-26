import dataclasses


@dataclasses.dataclass(frozen=True)
class StateResult:
    state: str
    winner: str
    frac_dem: float
    frac_rep: float
    frac_other: float
    total_votes: int
    from_beliefs: 'StateBeliefs' = None

    def get_dem_frac_of_major(self):
        return self.frac_dem / (self.frac_dem + self.frac_rep)

    def get_rep_frac_of_major(self):
        return self.frac_rep / (self.frac_dem + self.frac_rep)


@dataclasses.dataclass(frozen=True)
class StateBeliefs:
    state: str
    frac_dem_avg: float
    frac_rep_avg: float
    frac_other_avg: float
    total_votes: int
    dem_bump_from_new_candidate: float = 0
    dem_chaos_factor: float = 0
    weighted_poll_count: float = 0


@dataclasses.dataclass(frozen=True)
class Election:
    state_results: dict[str, StateResult]
    remaining_states: list[StateBeliefs]
    dem_candidate: str


