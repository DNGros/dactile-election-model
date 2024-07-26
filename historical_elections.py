import functools
from pprint import pprint
import requests

import pandas as pd
from pathlib import Path

from election_statics import DEM, REP, OTHER, get_all_state_codes, convert_state_name_to_state_code, BIDEN
from election_structs import Election, StateResult
cur_path = Path(__file__).parent.absolute()
data_path = cur_path / "mitelectiondata"

from joblib import Memory

cache = Memory(cur_path / "cache", verbose=1)


def _reformat_mit_election_data(df):
    # replace "DEMOCRAT" with "DEM"
    df["party"] = df["party"].replace("DEMOCRAT", DEM)
    # replace "REPUBLICAN" with "REP"
    df["party"] = df["party"].replace("REPUBLICAN", REP)
    # replace everything else with "OTHER"
    df["party"] = df["party"].apply(lambda x: OTHER if x not in [DEM, REP] else x)
    # rename the `state_po` to `state_code
    df = df.rename(columns={"state_po": "state_code"})
    return df


@functools.cache
def get_all_presidents_elections():
    df = pd.read_csv(data_path / "1976-2020-president.csv")
    df["party"] = df["party_detailed"]
    return _reformat_mit_election_data(df)


def get_2020_results():
    df = get_all_presidents_elections()
    df = df[df["year"] == 2020]
    return df


@functools.cache
def get_2020_election_struct() -> Election:
    df = get_2020_results()
    state_results = {}
    for state_code in get_all_state_codes():
        d = df[df["state_code"] == state_code]
        if len(d) == 0:
            continue
        dem_frac = d[d["party"] == DEM]["candidatevotes"].sum() / d["totalvotes"].max()
        rep_frac = d[d["party"] == REP]["candidatevotes"].sum() / d["totalvotes"].max()
        other_frac = d[d["party"] == OTHER]["candidatevotes"].sum() / d["totalvotes"].max()
        assert 1 - 1e-6 < dem_frac + rep_frac + other_frac <= 1 + 1e-6
        winner = DEM if dem_frac > rep_frac else REP
        state_results[state_code] = StateResult(state_code, winner, float(dem_frac), float(rep_frac), float(other_frac), total_votes=int(d["totalvotes"].mean()))
    return Election(state_results, [], dem_candidate=BIDEN)


def president_county_data(year=None):
    df = pd.read_csv(data_path / "countypres_2000-2020.csv")
    df = _reformat_mit_election_data(df)
    if year:
        df = df[df["year"] == year]
    df['state_county'] = df['state_code'] + "_" + df['county_name']
    return df


def michigan_county_governor_2022():
    url = "https://politics.api.cnn.io/results/county-races/2022-GG-MI.json"
    r = requests.get(url)
    r.raise_for_status()
    content = r.json()
    return cnn_proc_2022(content, "Michigan", 2022)


def cnn_proc_2018(content, state_name: str, year):
    data = []
    for county in content['counties']:
        pprint(county)
        other_data = dict(county)
        del other_data["race"]
        for candidate in county['race']["candidates"]:
            condidate_data = {
                **candidate,
                **other_data,
                "state": state_name,
                "state_code": convert_state_name_to_state_code(state_name),
                "year": year,
            }
            data.append(condidate_data)
    return pd.DataFrame(data)


def cnn_proc_2022(content, state_name: str, year):
    data = []
    for county in content:
        other_data = dict(county)
        del other_data["candidates"]
        for candidate in county["candidates"]:
            condidate_data = {
                **candidate,
                **other_data,
                "state": state_name,
                "state_code": convert_state_name_to_state_code(state_name),
                "year": year,
            }
            data.append(condidate_data)
    df = pd.DataFrame(data)
    return df


@functools.cache
@cache.cache()
def michigan_county_governor_2018():
    url = "https://data.cnn.com/ELECTION/2018November6/MI/county/G.json"
    r = requests.get(url)
    r.raise_for_status()
    content = r.json()
    return cnn_proc_2018(content, "Michigan", 2018)


def format_mit_county_data_into_simple_form(df):
    assert df["year"].nunique() == 1
    # Grab just the relevant columns into a new DataFrame
    df_relevant = df[["state_code", "county_name", "party", "candidatevotes", "totalvotes", "year"]]

    # Aggregate all party == "OTHER" when state_code and county_name are the same
    df_aggregated = df_relevant.groupby(["state_code", "county_name", "party"]).sum().unstack().fillna(0)

    # Flatten the MultiIndex columns
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]

    # Reset the index so that state_code and county_name are columns
    df_aggregated = df_aggregated.reset_index()

    # Add the state_county column
    df_aggregated['state_county'] = df_aggregated['state_code'] + "_" + df_aggregated['county_name']

    # Grab the unique county_fips for each state_code and county_name
    df_fips = df.drop_duplicates(subset=["state_code", "county_name"])[
        ["state_code", "county_name", "county_fips"]]

    # Merge the aggregated data with the county_fips
    df_final = pd.merge(df_aggregated, df_fips, on=["state_code", "county_name"], how="left")

    # Fill NaN values in county_fips with a default value and cast to int
    df_final['county_fips'] = df_final['county_fips'].fillna(0).astype(int)
    df_final["election_type"] = "president"

    # We actually only want one totalvotes column
    df_final["totalvotes"] = df_final["totalvotes_DEM"] + df_final["totalvotes_OTHER"] + df_final["totalvotes_REP"]
    df_final = df_final.drop(columns=["totalvotes_DEM", "totalvotes_OTHER", "totalvotes_REP"])
    # match the year
    df_final = df_final.drop(columns=["year_DEM", "year_OTHER"])
    df_final["year"] = df_final["year_REP"]
    df_final = df_final.drop(columns=["year_REP"])


    return df_final


def format_cnn_county_data_simple_form(df):
    print(df.columns)
    print(df.head)
    # rename the cnn cols
    if df['year'].max() == 2018: # 2018
        df = df.rename(columns={
            "votes": "candidatevotes",
            "name": "county_name",
            "id": "county_fips",
            "election_type": "gov",
        })
    elif df['year'].max() == 2022:
        df = df.rename(columns={
            "voteStr": "candidatevotes",
            "countyName": "county_name",
            "countyFipsCode": "county_fips",
            "election_type": "gov",
            "candidatePartyCode": "party",
        })
        # cast candidatevotes to int (replace "," with "")
        df["candidatevotes"] = df["candidatevotes"].str.replace(",", "").astype(int)
    else:
        raise RuntimeError

    # Select relevant columns
    df_relevant = df[["state_code", "county_name", "party", "candidatevotes", "county_fips", "year"]].copy()
    # asser one year
    assert df_relevant["year"].nunique() == 1
    year = df_relevant["year"].max()

    # Replace party with DEM, REP, OTHER
    party_mapping = {
        "D": "DEM",
        "R": "REP"
    }
    df_relevant["party"] = df_relevant["party"].map(party_mapping).fillna("OTHER")

    # Pivot the DataFrame
    df_pivoted = df_relevant.pivot_table(
        index=["state_code", "county_name"],
        columns="party",
        values="candidatevotes",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Rename the columns to match the MIT format
    df_pivoted.columns.name = None
    df_pivoted = df_pivoted.rename(columns={
        "DEM": "candidatevotes_DEM",
        "REP": "candidatevotes_REP",
        "OTHER": "candidatevotes_OTHER"
    })

    # Ensure all party columns exist
    for party in ['DEM', 'REP', 'OTHER']:
        if f"candidatevotes_{party}" not in df_pivoted.columns:
            df_pivoted[f"candidatevotes_{party}"] = 0

    # Calculate total votes
    df_pivoted["totalvotes"] = df_pivoted[["candidatevotes_DEM", "candidatevotes_REP", "candidatevotes_OTHER"]].sum(axis=1)

    # Make the county_name all uppercase
    df_pivoted["county_name"] = df_pivoted["county_name"].str.upper()

    df_pivoted["state_county"] = df_pivoted["state_code"] + "_" + df_pivoted["county_name"]
    df_pivoted["election_type"] = "gov"

    # Add county_fips back to the dataframe
    county_fips = df_relevant.groupby(["state_code", "county_name"])["county_fips"].first().reset_index()
    df_final = pd.merge(df_pivoted, county_fips, on=["state_code", "county_name"], how="left")

    # Reorder columns to match MIT format
    column_order = [
        "state_code", "county_name", "candidatevotes_DEM", "candidatevotes_OTHER", "candidatevotes_REP",
        "totalvotes", "state_county", "county_fips", "election_type"
    ]
    df_final = df_final.reindex(columns=column_order)
    df_final["year"] = year
    return df_final


if __name__ == "__main__":
    with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
        print(get_2020_results())
        exit()
        #michigan_county_data()
        #print(michigan_county_governor_2022())
        df = president_county_data()
        print("Pres col")
        df = df[df["year"] == 2020]
        format_national_df = format_mit_county_data_into_simple_form(df)
        #df = michigan_county_governor_2018()
        df = michigan_county_governor_2022()
        format_michigan_df = format_cnn_county_data_simple_form(df)
        print("Gov col")
        print(df.columns)
        print("pres col format")
        print(format_national_df.columns)
        raise RuntimeError
        print("gov col format")
        print(format_michigan_df.columns)
        # Assert cols the same
        if not (set(format_national_df.columns) == set(format_michigan_df.columns)):
            print(set(format_national_df.columns) - set(format_michigan_df.columns))
            print(set(format_michigan_df.columns) - set(format_national_df.columns))
            raise RuntimeError
        print(format_national_df.head())
        print(format_michigan_df.head())
        exit()
    # limit to no max display columns or text per line
    with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
        print(get_all_presidents_elections().columns)
        print(get_2020_results().state.unique())
        print(get_2020_results().state_cen.unique())
        print(get_2020_results().state_ic.unique())
        print(get_2020_results().state_code.unique())
    print(get_2020_election_struct())