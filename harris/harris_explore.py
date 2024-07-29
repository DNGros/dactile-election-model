import functools
from typing import Literal
from collections import defaultdict
from pathlib import Path

from election_statics import convert_state_name_to_state_code, HARRIS, BIDEN
from get_poll_data import get_pres_data_from_csv, get_pres_data_from_csv_past_cycles
import pandas as pd

from hyperparams import default_poll_time_penalty, swing_states
from util import pd_all_columns, pd_all_rows
import numpy as np

cur_path = Path(__file__).parent.absolute()

useful_col = ['poll_id', 'numeric_grade', 'state', 'start_date', 'end_date', 'candidate_name', 'question_id',
               'pct', 'pollster', 'url', 'question_has_biden', 'question_has_harris', 'after_dropout']


@functools.cache
def get_avail_harris_biden_polls():
    return get_avail_filtered(mode="HARRIS")


def get_avail_filtered(
    start_date: pd.Timestamp = pd.Timestamp('2024-06-28'),
    mode: str = "HARRIS",
    cycle: int = 2024,
):
    if cycle == 2024:
        df = get_pres_data_from_csv()
    elif cycle == 2020:
        df = get_pres_data_from_csv_past_cycles()
    else:
        raise ValueError()
    #print(df.columns)
    #with pd.option_context('display.max_columns', None, 'display.width', None):
    #    print(df.head())
    """
    Index(['poll_id', 'pollster_id', 'pollster', 'sponsor_ids', 'sponsors',
       'display_name', 'pollster_rating_id', 'pollster_rating_name',
       'numeric_grade', 'pollscore', 'methodology', 'transparency_score',
       'state', 'start_date', 'end_date', 'sponsor_candidate_id',
       'sponsor_candidate', 'sponsor_candidate_party', 'endorsed_candidate_id',
       'endorsed_candidate_name', 'endorsed_candidate_party', 'question_id',
       'sample_size', 'population', 'subpopulation', 'population_full',
       'tracking', 'created_at', 'notes', 'url', 'source', 'internal',
       'partisan', 'race_id', 'cycle', 'office_type', 'seat_number',
       'seat_name', 'election_date', 'stage', 'nationwide_batch',
       'ranked_choice_reallocated', 'ranked_choice_round', 'party', 'answer',
       'candidate_id', 'candidate_name', 'pct'],
      dtype='object')
   poll_id  pollster_id pollster sponsor_ids sponsors     display_name  pollster_rating_id pollster_rating_name  numeric_grade  pollscore                   methodology  transparency_score state start_date end_date  sponsor_candidate_id sponsor_candidate sponsor_candidate_party  endorsed_candidate_id  endorsed_candidate_name  endorsed_candidate_party  question_id  sample_size population  subpopulation population_full tracking    created_at notes                                                url  source internal partisan  race_id  cycle     office_type  seat_number  seat_name election_date    stage  nationwide_batch  ranked_choice_reallocated  ranked_choice_round party   answer  candidate_id     candidate_name   pct
0    87361         1102  Emerson         NaN      NaN  Emerson College                  88      Emerson College            2.9       -1.1  IVR/Online Panel/Text-to-Web                 7.0   NaN     7/7/24   7/8/24                   NaN               NaN                     NaN                    NaN                      NaN                       NaN       202461       1370.0         rv            NaN              rv      NaN  7/9/24 11:36   NaN  https://emersoncollegepolling.com/july-2024-na...     NaN      NaN      NaN     8914   2024  U.S. President            0        NaN       11/5/24  general             False                      False                  NaN   DEM    Biden         19368          Joe Biden  49.8
1    87361         1102  Emerson         NaN      NaN  Emerson College                  88      Emerson College            2.9       -1.1  IVR/Online Panel/Text-to-Web                 7.0   NaN     7/7/24   7/8/24                   NaN               NaN                     NaN                    NaN                      NaN                       NaN       202461       1370.0         rv            NaN              rv      NaN  7/9/24 11:36   NaN  https://emersoncollegepolling.com/july-2024-na...     NaN      NaN      NaN     8914   2024  U.S. President            0        NaN       11/5/24  general             False                      False                  NaN   REP    Trump         16651       Donald Trump  50.2
2    87361         1102  Emerson         NaN      NaN  Emerson College                  88      Emerson College            2.9       -1.1  IVR/Online Panel/Text-to-Web                 7.0   NaN     7/7/24   7/8/24                   NaN               NaN                     NaN                    NaN                      NaN                       NaN       202462       1370.0         rv            NaN              rv      NaN  7/9/24 11:36   NaN  https://emersoncollegepolling.com/july-2024-na...     NaN      NaN      NaN     8914   2024  U.S. President            0        NaN       11/5/24  general             False                      False                  NaN   DEM    Biden         19368          Joe Biden  39.9
3    87361         1102  Emerson         NaN      NaN  Emerson College                  88      Emerson College            2.9       -1.1  IVR/Online Panel/Text-to-Web                 7.0   NaN     7/7/24   7/8/24                   NaN               NaN                     NaN                    NaN                      NaN                       NaN       202462       1370.0         rv            NaN              rv      NaN  7/9/24 11:36   NaN  https://emersoncollegepolling.com/july-2024-na...     NaN      NaN      NaN     8914   2024  U.S. President            0        NaN       11/5/24  general             False                      False                  NaN   REP    Trump         16651       Donald Trump  43.7
4    87361         1102  Emerson         NaN      NaN  Emerson College                  88      Emerson College            2.9       -1.1  IVR/Online Panel/Text-to-Web                 7.0   NaN     7/7/24   7/8/24                   NaN               NaN                     NaN                    NaN                      NaN                       NaN       202462       1370.0         rv            NaN              rv      NaN  7/9/24 11:36   NaN  https://emersoncollegepolling.com/july-2024-na...     NaN      NaN      NaN     8914   2024  U.S. President            0        NaN       11/5/24  general             False                      False                  NaN   IND  Kennedy         31042  Robert F. Kennedy   6.1
    """
    # Filter poll_id to polls that contain Harris
    if mode == HARRIS:
        harris_rows = df[df['candidate_name'].str.lower().str.contains('harris')]
        df = df[df.poll_id.isin(harris_rows.poll_id)].copy()
    elif mode == BIDEN:
        biden_rows = df[df['candidate_name'].str.lower().str.contains('biden')]
        df = df[df.poll_id.isin(biden_rows.poll_id)].copy()
    elif mode == "BOTH":
        biden_rows = df[df['candidate_name'].str.lower().str.contains('harris|biden|trump')]
        df = df[df.poll_id.isin(biden_rows.poll_id)].copy()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    def group_has_candidate(group, candidate):
        return candidate in ' '.join(group.str.lower()).split()

    df['question_has_biden'] = df.groupby('question_id')['candidate_name'].transform(group_has_candidate,
                                                                                     'biden')
    df['question_has_harris'] = df.groupby('question_id')['candidate_name'].transform(group_has_candidate,
                                                                                      'harris')
    df['is_major_candidate'] = df['candidate_name'].str.lower().str.contains('biden|harris|trump')
    if mode == HARRIS:
        q_ids = df[df['candidate_name'].str.lower().str.contains('harris')].question_id
    elif mode == BIDEN:
        q_ids = df[df['candidate_name'].str.lower().str.contains('biden')].question_id
    else:
        q_ids = df[df['candidate_name'].str.lower().str.contains('harris|biden')].question_id
    df = df[df.question_id.isin(q_ids)]
    # convert the dates to datetime
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce', format='%m/%d/%y')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce', format='%m/%d/%y')
    # Filter dates
    if start_date is not None:
        df = df[df['start_date'] >= start_date]
    df['state_code'] = df['state'].apply(convert_state_name_to_state_code)
    df['after_dropout'] = df['start_date'] > pd.to_datetime('2024-07-21')
    return df


def explore_avail_biden_harris():
    df = get_avail_harris_biden_polls()
    with pd.option_context(*pd_all_columns, *pd_all_rows, 'display.max_colwidth', None):
        # print indexed / grouped by ['poll_id', 'question_id']
        print(df[useful_col].set_index(['poll_id', 'question_id']).sort_index())


def explore_avail_harris():
    df = get_avail_filtered(mode=HARRIS)
    with pd.option_context(*pd_all_columns, *pd_all_rows, 'display.max_colwidth', None):
        # print indexed / grouped by ['poll_id', 'question_id']
        print(df[useful_col].set_index(['poll_id', 'question_id']).sort_index())


def poll_weight(
    numeric_grade,
    is_projection,
    start_date,
    end_date,
    pollscore,
    sample_size,
    reference_today_date,
    time_penalty=default_poll_time_penalty,
    harris_and_before_dropout=False,
    pollster_name: str = None,
    exclude_pollsters: list[str] | None = None,  # To answer a reddit question, not actually used
):
    """Given factors about a poll return some weighting. This is somewhat
    arbitrary and could be adjusted.

    Args:
        numeric_grade: The 538 rating of the pollster. Rated out of 3.0.
        is_projection: If the poll is a projection (not currently used)
        start_date: The start date of the poll
        end_date: The end date of the poll
        pollscore: The 538 pollster score. Lower values are better. Like the
            numberic grade, but just measures empirical track record, not
            factors like transparency.
        sample_size: The number of people polled
        reference_today_date: The date to use when weighting old polls
        time_penalty: Approximately the days half life
        harris_and_before_dropout: If the poll is for Harris and before the
            dropout day
        pollster_name: The name of the pollster
        exclude_pollsters: A list of pollsters to exclude (matches any substring)
            Not actually used in the function, but is a parameter to make it
            easier to answer a reddit question.
    """
    # If NaN then numeric_grade 1.5
    if pd.isna(numeric_grade):
        grade_clipped = 1.5
    else:
        grade_clipped = max(0.0, min(2.7, numeric_grade))
    score = grade_clipped**1.5 / 2.7**1.5
    if is_projection:
        score = (score**2) * 0.5
    # Pollscore which is some value considering only performance (ignoring transparency)
    # Lower values are better
    if pd.isna(pollscore):
        pollscore = 0.0
    score *= np.interp(pollscore, [-1.1, 0.5], [1.0, 0.6])
    # Some sample size consideration
    score *= np.interp(sample_size, [200, 900], [0.7, 1.0])
    # Time decay
    end_days = (reference_today_date - end_date).days
    if end_days < 0:
        return 0  # in the future
    start_days = (reference_today_date - start_date).days
    # Find a middle date (putting more weight on end date since I think some
    # pollsters will start out smaller and scale up(??))
    days = (end_days * 2 + start_days * 1) / 3
    days -= 1.5  # Prep time
    days = max(0, days)
    # Subtract a prep day
    time_decay = 0.5 ** (days / time_penalty)
    if harris_and_before_dropout:
        score *= 0.25
    score *= time_decay
    # Especially punish low quality
    #score = score ** 1 + days.days / time_decay

    if exclude_pollsters and pollster_name: # for reddit question
        for pollster in exclude_pollsters:
            if pollster_name.lower() in pollster.lower():
                return 0.0

    if score < 0:
        return 0.0
    return score


def build_harris_national_table_df():
    return build_polls_clean_df(mode='BOTH')


@functools.cache
def build_polls_clean_df(
    mode: Literal['HARRIS', 'BIDEN', 'BOTH'],
    state=None,
    cycle: int = 2024,
    reference_date=pd.Timestamp.now(),
    start_date=pd.Timestamp('2024-06-28')
):
    has_harris = mode in [HARRIS, "BOTH"]
    has_biden = mode in [BIDEN, 'BOTH']
    assert has_harris or has_biden
    if start_date.year > cycle:
        raise ValueError("start_date is after cycle")
    df = get_avail_filtered(
        mode=mode,
        start_date=start_date,
        cycle=cycle,
    )
    # Filter to national polls
    if state is None:
        df = df[df['state'].isna()]
    else:
        df = df[df['state_code'] == convert_state_name_to_state_code(state)]
    if len(df) == 0:
        return None
    # There will be rows with the same poll_id that multiple questions
    # with both Trump and Biden. We want to take the question_id that
    # has the largest sum
    df['question_major_candidate_sum'] = df[df['is_major_candidate']].groupby('question_id')['pct'].transform('sum')

    # Function to get the top question for a candidate
    def get_top_question(group):
        return group.loc[group['question_major_candidate_sum'].idxmax()]

    # Get the top questions for Biden and Harris separately
    top_biden = df[df['question_has_biden']].groupby('poll_id').apply(get_top_question).reset_index(drop=True)
    top_harris = df[df['question_has_harris']].groupby('poll_id').apply(get_top_question).reset_index(drop=True)

    # filter df to questions that are in top_biden or top_harris
    df = df[df['question_id'].isin(top_biden['question_id']) | df['question_id'].isin(top_harris['question_id'])]

    # Combine the results and remove None values
    df = df.dropna(subset=['poll_id'])  # Remove rows where poll_id is None

    # Reset index and sort
    df = df.reset_index(drop=True)
    df = df.sort_values(['end_date', 'poll_id'], ascending=[False, True])

    df = df[df['is_major_candidate']]

    if mode == HARRIS:
        df = df[df['question_has_harris']]
    elif mode == BIDEN:
        df = df[df['question_has_biden']]

    #df = df[df['pollster'] == "Siena/NYT"]
    df = df.sort_values('end_date', ascending=False)

    vals = []
    for poll_id, group in df.groupby('poll_id'):
        biden_and_trump = group[group['question_has_biden']]
        harris_and_trump = group[group['question_has_harris']]
        biden = biden_and_trump[biden_and_trump['candidate_name'].str.lower().str.contains('biden')]
        trump_v_biden = biden_and_trump[biden_and_trump['candidate_name'].str.lower().str.contains('trump')]
        harris = harris_and_trump[harris_and_trump['candidate_name'].str.lower().str.contains('harris')]
        trump_v_harris = harris_and_trump[harris_and_trump['candidate_name'].str.lower().str.contains('trump')]
        row_vals = {
            'poll_id': poll_id,
            'pollster': group['pollster'].values[0],
            'url': group['url'].values[0],
            'numeric_grade': group['numeric_grade'].values[0],
            'pollscore': group['pollscore'].values[0],
            'start_date': group['start_date'].values[0],
            'end_date': group['end_date'].values[0],
            'sample_size': group['sample_size'].values[0],
            'custom_weight': poll_weight(
                group['numeric_grade'].values[0],
                False,
                group['start_date'].values[0],
                group['end_date'].values[0],
                group['pollscore'].values[0],
                group['sample_size'].values[0],
                reference_date,
                harris_and_before_dropout=(
                    mode == HARRIS
                    and not group['after_dropout'].values[0]
                ),
                pollster_name=group['pollster'].values[0],
            ),
        }
        if has_harris:
            if len(harris_and_trump) != 2:
                continue
            harris_pct = harris['pct'].values[0]
            harris_frac = harris_pct / harris_and_trump['pct'].sum()
            row_vals |= {
                'harris_pct': harris_pct,
                'trump_v_harris_pct': trump_v_harris['pct'].values[0],
                'harris_frac': harris_frac,
            }
        if has_biden:
            if len(biden_and_trump) != 2:
                continue
            biden_pct = biden['pct'].values[0]
            biden_frac = biden_pct / biden_and_trump['pct'].sum()
            row_vals |= {
                'biden_pct': biden_pct,
                'trump_v_biden_pct': trump_v_biden['pct'].values[0],
                'biden_frac': biden_frac,
            }
        if has_biden and has_harris:
            row_vals |= {
                'harris_delta': harris_frac - biden_frac,
            }
        vals.append(row_vals)
    table_df = pd.DataFrame(vals)
    #with pd.option_context(*pd_all_columns, *pd_all_rows, 'display.max_colwidth', None):
    #    print(table_df.set_index('poll_id'))
    #    print("Num polls", len(vals))
    #    print("Avg Harris Delta:", table_df['harris_delta'].mean())
    #    print("Harris delta std:", table_df['harris_delta'].std())
    #    weighted_mean = np.average(table_df['harris_delta'], weights=table_df['custom_weight'])
    #    print("Weighted Avg Harris Delta:", weighted_mean)
    #    weighted_variance = np.average((table_df['harris_delta'] - weighted_mean) ** 2, weights=table_df['custom_weight'])
    #    weighted_std_dev = np.sqrt(weighted_variance)
    #    print("Weighted Harris delta std:", weighted_std_dev)
    return table_df


@functools.cache
def get_average_harris_delta(weighted: bool = False):
    df = build_harris_national_table_df()
    if weighted:
        return np.average(df['harris_delta'], weights=df['custom_weight'])
    else:
        return df['harris_delta'].mean()

def harris_national_table_html():
    df = build_harris_national_table_df()
    df = df.sort_values('end_date', ascending=False)
    lines = []
    lines.append("<table>")
    lines.append("<tr>")
    for col in [
        "Pollster (538's <a href='https://projects.fivethirtyeight.com/pollster-ratings/'>rating</a>. 3 max)",
        "Dates",
        "Biden | Trump",
        "Harris | Trump",
        "Biden Frac",
        "Harris Frac",
        "Harris Delta Points",
    ]:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr>")
    for _, row in df.iterrows():
        lines.append("<tr>")
        numeric_grade = row['numeric_grade']
        if pd.isna(numeric_grade):
            numeric_grade = "?"
        else:
            numeric_grade = f"{numeric_grade:.1f}"
        for col in [
            f'<a href="{row["url"]}">{row["pollster"]}</a> ({numeric_grade})',
            f"{row['start_date'].strftime('%m/%d')}-{row['end_date'].strftime('%m/%d')}",
            f"{row['biden_pct']:.0f}% | {row['trump_v_biden_pct']:.0f}%",
            f"{row['harris_pct']:.0f}% | {row['trump_v_harris_pct']:.0f}%",
            f"{row['biden_frac']:.3f}",
            f"{row['harris_frac']:.3f}",
            f"{row['harris_delta']*100:+.1f}",
        ]:
            lines.append(f"<td>{col}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return '\n'.join(lines)


harris_swing_states = swing_states + ['VA', 'TX']

def get_extra_polls():
    return {
        "OpenLab ??": {
            "url": {}
        }
    }

def build_harris_swing_state():
    swing_dfs = []
    for state in harris_swing_states:
        state_code = convert_state_name_to_state_code(state)
        df = build_polls_clean_df("BOTH", state_code)
        if df is None:
            continue
        df['state_code'] = state_code
        print("State:", state)
        with pd.option_context('display.max_columns', None, 'display.width', None):
            print(df)
        swing_dfs.append(df)
    df = pd.concat(swing_dfs)
    # Group by a combination of the pollster and the end_date
    df['pollster_end_date'] = df['pollster'] + df['end_date'].dt.strftime(' %m/%d')
    pollster_to_state_to_vals = {}
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(df.set_index('pollster_end_date').sort_index())
    for pollster_end_date, group in df.groupby('pollster_end_date'):
        pollster_to_state_to_vals[pollster_end_date] = group.set_index('state_code').to_dict()
    print(pollster_to_state_to_vals)
    return pollster_to_state_to_vals


@functools.cache
def harris_swing_state_table_html():
    pollster_to_state_to_vals = build_harris_swing_state()
    lines = []
    lines.append("<table>")
    lines.append("<tr>")
    for col in [
        "Pollster (<a href='https://projects.fivethirtyeight.com/pollster-ratings/'>rating</a>)",
        "Weight",
        *map(convert_state_name_to_state_code, harris_swing_states),
    ]:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr>")
    state_vals = defaultdict(list)
    for pollster, vals in pollster_to_state_to_vals.items():
        url = list(vals['url'].values())[0]
        grade = list(vals['numeric_grade'].values())[0]
        if pd.isna(grade):
            grade = "?"
        else:
            grade = f"{grade:.1f}"
        weight = list(vals['custom_weight'].values())[0]
        lines.append("<tr>")
        lines.append(f'<td><a href="{url}">{pollster}</a> ({grade})</td>')
        lines.append(f"<td>{weight:.2f}</td>")
        for state in harris_swing_states:
            state = convert_state_name_to_state_code(state)
            if state in vals['poll_id']:
                delta = vals['harris_delta'][state]
                lines.append(f"<td>{delta*100:+.1f}</td>")
                state_vals[state].append((delta, weight))
            else:
                lines.append("<td></td>")
    # Add the national average
    lines.append("<tr>")
    lines.append("<td>National Average</td>")
    national_weight = 1.0
    lines.append(f"<td>{national_weight:.1f}</td>")
    national_average = get_average_harris_delta(weighted=True)
    for state in harris_swing_states:
        state = convert_state_name_to_state_code(state)
        lines.append(f"<td>{national_average*100:+.1f}</td>")
        state_vals[state].append((national_average, national_weight))
    # Add a horizontal line
    #lines.append("<tr>")
    #lines.append("<td colspan='100%'><hr></td>")
    #lines.append("</tr>")
    # Add the state averages
    lines.append("<tr>")
    lines.append("<td>State Weighted Mean</td>")
    lines.append("<td></td>")
    print(state_vals)
    swing_state_to_avg = {}
    for state in harris_swing_states:
        state = convert_state_name_to_state_code(state)
        deltas, weights = zip(*state_vals[state])
        state_average = np.average(deltas, weights=weights)
        lines.append(f"<td>{state_average*100:+.1f}</td>")
        swing_state_to_avg[state] = state_average
    swing_state_to_avg[None] = national_average
    # Add the state standard deviations
    #lines.append("<tr>")
    #lines.append("<td>Weighted SD</td>")
    #lines.append("<td></td>")
    #for state in harris_swing_states:
    #    state = convert_state_name_to_state_code(state)
    #    deltas, weights = zip(*state_vals[state])
    #    state_average = np.average(deltas, weights=weights)
    #    state_variance = np.average((np.array(deltas) - state_average)**2, weights=weights)
    #    state_std_dev = np.sqrt(state_variance)
    #    weights_sum = sum(weights)
    #    state_sem = (state_std_dev * 100) / np.sqrt(weights_sum)
    #    lines.append(f"<td>{state_std_dev*100:.1f}:{state_sem:.1f}</td>")
    #lines.append("</tr>")
    lines.append("</table>")
    return '\n'.join(lines), swing_state_to_avg


if __name__ == "__main__":
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_rows', None):
        print(get_pres_data_from_csv().head(100))
    #explore_avail_biden_harris()
    #explore_avail_harris()
    #build_harris_national_table_df()
    #Jj:w
    # print(harris_on())

    #table = harris_only_table_html()
    #toy_html = ["<html><head><title>Toy Harris</title></head><body>"]
    #toy_html.append(table)
    #toy_html.append("</body></html>")
    #with open(cur_path / "seetable.html", "w") as f:
    #    f.write('\n'.join(toy_html))
    pass
