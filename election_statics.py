import requests
import pandas as pd
from bs4 import BeautifulSoup

from hyperparams import swing_states

DEM = "DEM"
REP = "REP"
OTHER = "OTHER"

BIDEN = "BIDEN"
WHITMER = "WHITMER"
HARRIS = "HARRIS"


def get_electoral_votes() -> tuple[pd.DataFrame, dict[str, int]]:
    import pandas as pd

    data = {
        "state": ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
                         "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho",
                         "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
                         "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
                         "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
                         "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
                         "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
                         "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
                         "Wyoming"],
        "votes": [9, 3, 11, 6, 54, 10, 7, 3, 3, 30, 16, 4, 4, 19, 11, 6, 6, 8, 8, 4, 10, 11, 15, 10,
                            6, 10, 4, 5, 6, 4, 14, 5, 28, 16, 3, 17, 7, 8, 19, 4, 9, 3, 11, 40, 6, 3, 13, 12,
                            4, 10, 3],
        "Change from 2020": [0, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1,
                                      0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0, -1, 0, 1, -1, 0, 0, 0, 0, 2, 0, 0,
                                      0, 0, -1, 0, 0]
    }

    df = pd.DataFrame(data)
    data['state_code'] = [convert_state_name_to_state_code(state) for state in data['state']]
    state_to_electors = dict(zip(data['state_code'], data['votes']))
    return df, state_to_electors



_state_dict = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY"
}


def get_all_state_codes() -> list[str]:
    """Returns a list of all state codes"""
    return list(_state_dict.values())


def convert_state_name_to_state_code(state: str) -> str:
    """Converts the state to a two letter state name"""
    if isinstance(state, float):
        return None
    if len(state) == 2:
        return state.upper()
    state = state.replace("-", " ").lower()
    return _state_dict.get(state)


_state_code_to_name = {
    convert_state_name_to_state_code(state): state
    for state in swing_states
}


def state_code_to_state_name(state_code) -> str | None:
    if state_code is None:
        return None
    return _state_code_to_name.get(state_code)
