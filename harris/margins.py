import functools

import pandas as pd

from election_statics import convert_state_name_to_state_code
from get_poll_data import get_poll_averages_from_csv
from hyperparams import swing_states


@functools.cache
def get_margins_df(only_swing_states: bool):
    """A df of poll averages with information about the win margins"""
    df = get_poll_averages_from_csv()
    print(df.columns)
    # Index(['candidate', 'date', 'pct_trend_adjusted', 'state', 'cycle', 'party',
    #        'pct_estimate', 'hi', 'lo'],
    #       dtype='object')
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_rows', 4):
        print(df)
        #                  candidate        date  pct_trend_adjusted      state  cycle party  pct_estimate         hi         lo
        # 0      Joseph R. Biden Jr.  2020-11-03            37.82732    Alabama   2020   NaN           NaN        NaN        NaN
        # 1             Donald Trump  2020-11-03            57.36126    Alabama   2020   NaN           NaN        NaN        NaN
        # ...                    ...         ...                 ...        ...    ...   ...           ...        ...        ...
        # 25807              Kennedy  2024-03-01                 NaN  Wisconsin   2024   IND       10.3090  13.044215   7.668578
        # 25808                Biden  2024-03-01                 NaN  Wisconsin   2024   DEM       38.8528  41.241032  36.374370


    # parse the date
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    # Filter to March 1 since that's when the 2024 data starts
    df = df[df['day_of_year'] >= 60].copy()
    # pct which is either the pct_estimate or pct_trend_adjusted (whichever is not NaN)
    df.loc[:, 'pct'] = df['pct_estimate'].fillna(df['pct_trend_adjusted'])
    # Filter to just Biden and Trump (case insensitive)
    df = df[df['candidate'].str.lower().str.contains('biden|trump')]
    # filter to 2024
    #df = df[df['date'].str.contains('2024')]
    # Get the total pct for each (date, state)
    df['total_pct'] = df.groupby(['date', 'state'])['pct'].transform('sum')
    df['percent_of_d_r_share'] = df['pct'] / df['total_pct'] * 100
    # Just have biden
    df = df[df['candidate'].str.lower().str.contains('biden')]
    df['d_r_share_margin'] = df['percent_of_d_r_share'] - 50
    if only_swing_states:
        df = df[df['state'].isin(swing_states)]
    df['state_code'] = df['state'].apply(convert_state_name_to_state_code)
    return df
