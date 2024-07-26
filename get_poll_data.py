import requests
import pandas as pd
import functools
import json
import logging
from joblib import Memory

from election_statics import convert_state_name_to_state_code
from pathlib import Path

# Set up logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up joblib caching
cur_path = Path(__file__).parent.absolute()
cache_dir = cur_path / "cache"
memory = Memory(cache_dir, verbose=1)
import datetime
#memory.reduce_size(age_limit=datetime.timedelta(hours=6))

def combine_daily_polls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows from the same day into a single row, creating new columns for each candidate.

    Args:
    df (pd.DataFrame): The original DataFrame with separate rows for each candidate.

    Returns:
    pd.DataFrame: A DataFrame with one row per day and columns for each candidate's metrics.
    """
    # replace "IND" with "OTHER"
    df['party'] = df['party'].replace("IND", "OTHER")
    # Pivot the DataFrame to create columns for each candidate's metrics
    pivoted = df.pivot(index='date', columns='party', values=['pct_estimate', 'hi', 'lo'])

    # Flatten the column names
    pivoted.columns = [f'{candidate}_{metric}' for metric, candidate in pivoted.columns]

    # Reset the index to make 'date' a column again
    pivoted = pivoted.reset_index()

    return pivoted


def get_national_averages() -> pd.DataFrame:
    """
    Get the national polling averages for the 2024 presidential election.

    Returns:
    pd.DataFrame: A DataFrame containing the combined national polling averages.
    """
    url = "https://projects.fivethirtyeight.com/polls/president-general/2024/national/polling-average.json"
    return get_poll_averages(url, state="national")


@memory.cache()
def get_poll_averages(url: str, state: str = None) -> pd.DataFrame | None:
    """
    Download poll data from the given URL, convert it to a DataFrame, and combine daily polls.

    Args:
    url (str): The URL of the JSON data to download.
    state (str, optional): The name of the state for state-level polls.

    Returns:
    pd.DataFrame: A DataFrame containing the combined poll data.
    """
    logging.log(logging.INFO, f"Fetching data from {url}")
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = json.loads(response.text)
    df = pd.DataFrame(data)


    # Combine daily polls
    combined_df = combine_daily_polls(df)

    # Add state column if it's a state-level poll
    if state:
        combined_df['state'] = state

    return combined_df


@functools.cache
def get_state_averages() -> dict:
    """
    Get the state-level polling averages for the 2024 presidential election.
    Skips states that don't have data available.

    Returns:
    dict: A dictionary mapping state names to their respective DataFrames.
    """
    base_url = "https://projects.fivethirtyeight.com/polls/president-general/2024/{}/polling-average.json"
    states = [
        "michigan", "pennsylvania", "wisconsin", "florida", "north-carolina", "arizona", "georgia", "ohio",
        "iowa",
        "new-hampshire", "nevada", "colorado", "minnesota", "virginia", "maine", "new-mexico", "texas"
    ]  # Add more states as needed

    state_data = {}
    for state in states:
        state_code = convert_state_name_to_state_code(state)
        url = base_url.format(state)
        try:
            state_df = get_poll_averages(url, state)
            if state_df is not None and not state_df.empty:
                state_data[state_code] = state_df
            else:
                print(f"No data available for {state}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {state}: {e}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON data for {state}")
        except Exception as e:
            print(f"Unexpected error occurred for {state}: {e}")

    return state_data


@functools.cache
@memory.cache()
def get_pres_data_from_csv() -> pd.DataFrame:
    """Get president poll data from the download link"""
    url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
    #response = requests.get(url)
    #if response.status_code != 200:
    #    return None
    #print(response.text)
    # convert the csv to a dataframe
    df = pd.read_csv(url)
    return df


@functools.cache
@memory.cache()
def get_pres_data_from_csv_past_cycles() -> pd.DataFrame:
    url = "https://projects.fivethirtyeight.com/polls/data/president_polls_historical.csv"
    df = pd.read_csv(url)
    return df



@functools.cache
@memory.cache()
def get_poll_averages_from_csv() -> pd.DataFrame:
    url = "https://projects.fivethirtyeight.com/polls/data/presidential_general_averages.csv"
    df = pd.read_csv(url)
    return df


if __name__ == "__main__":
    df = get_pres_data_from_csv_past_cycles()
    print(df.columns)
    print(df.cycle.unique())
    #print(get_poll_averages_from_csv().candidate.unique())
    exit()
    df = get_pres_data_from_csv()
    print(df.columns)
    print(df['candidate_name'].unique())
    harris_df = df[df['candidate_name'].str.lower().str.contains("harris")]
    print(harris_df)
    exit()

    print(get_poll_averages("https://projects.fivethirtyeight.com/polls/president-general/2024/new-hampshire/polls.json", "new-hampshire"))
    exit()
    print(list(get_state_averages().keys()))
    exit()
    print("National Polling Averages:")
    national_averages = get_national_averages()
    print(national_averages.head())

    print("\nState-level Polling Averages:")
    state_averages = get_state_averages()
    for state, df in state_averages.items():
        print(f"\n{state.capitalize()}:")
        print(df.head())

    print(f"\nTotal number of states with data: {len(state_averages)}")

    # Optionally, save the DataFrames to CSV files
    # national_averages.to_csv("national_polling_averages.csv", index=False)
    # for state, df in state_averages.items():
    #     df.to_csv(f"{state}_polling_averages.csv", index=False)