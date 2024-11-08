import pandas as pd
from markupsafe import Markup

from daily.custom_poll_avg import find_national_avg_scaled_for_state
from daily.past_predictions import generate_html_calendar, read_all_past_vars, read_all_top_lines
from daily.vp import make_vp_shift_table, get_particular_value_state_shift
from election_statics import HARRIS
from harris.harris_explore import build_polls_clean_df

from hyperparams import swing_states


def make_all_daily_tables(article_vars):
    current_top_line = article_vars['top_line_prob']
    return {
        "harris_national_table": Markup(harris_only_table_html()),
        "swing_state_tables": {
            state: Markup(harris_only_table_html(state))
            for state in swing_states
        },
        "calendar": Markup(generate_html_calendar(
            read_all_top_lines(),
            today_value=current_top_line,
        )),
        # VP stuff
        #"vp_pa_shift_table": Markup(make_vp_shift_table("PA")),
        #"vp_az_shift_table": Markup(make_vp_shift_table("AZ")),
        ###"vp_pa_correlated_shift_table": Markup(make_vp_shift_table("PA", corr_factor=3)),
        #"vp_pa_shift_1_corr_3": get_particular_value_state_shift(
        #    "PA",
        #    mean=0.01,
        #    std=0.01,
        #    corr_dampening=3,
        #),
        #"vp_pa_shift_1": get_particular_value_state_shift(
        #    "PA",
        #    mean=0.01,
        #    std=0.01,
        #)
    }


def harris_only_table_html(
    state: str = None,
    include_national: bool = True,
):
    """Creates a table for displaying avaialble Harris polls."""
    df = build_polls_clean_df(mode=HARRIS, state=state)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #    print(df)
    #    exit()
    # Remove older than 60 days from largest date
    df = df[df['end_date'] >= df['end_date'].max() - pd.Timedelta(days=60)]
    df['national_row'] = False
    if state is not None and include_national:
        value, slope, intercept, weight = (
            find_national_avg_scaled_for_state(state, pd.Timestamp.now(), candidate=HARRIS))
        new_row = {
            'pollster': f'From Natl. Avg. ({slope:.2f}⋅x + {intercept:.2f})',
            'custom_weight': weight,
            'harris_frac': value / 100,
            'start_date': df['start_date'].min(),
            'end_date': df['end_date'].max(),
            'national_row': True,
        }
        df = pd.concat([pd.DataFrame([new_row]), df])
        #print("CONNNATJj:w")
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        #    print(df)
    df = df.sort_values('custom_weight', ascending=False)
    lines = []
    lines.append("<table class='gen_polls_table'>")
    lines.append("<thead>")
    lines.append("<tr>")

    # Same as above but manual
    lines.append('<th data-order="desc">Weight</th>')
    lines.append('<th data-dt-order="disable">Pollster (<a href="https://projects.fivethirtyeight.com/pollster-ratings/">rating</a>)</th>')
    lines.append("<th>Dates</th>")
    lines.append('<th data-dt-order="disable">Harris: Trump</th>')
    lines.append("<th>Harris Share</th>")

    lines.append("</tr>")
    lines.append("</thead>")
    lines.append("<tbody>")
    for _, row in df.iterrows():
        lines.append("<tr>")
        numeric_grade = row['numeric_grade']
        if pd.isna(numeric_grade):
            numeric_grade = "?"
        else:
            numeric_grade = f"{numeric_grade:.1f}"

        if f"{row['custom_weight']:.2f}" == "nan":
            print("NAN WEIGHT")
            print(row)
            raise ValueError("NAN WEIGHT")
        lines.append(f"<td>{row['custom_weight']:.2f}</td>")
        if not row.get('national_row', False):
            lines.append(f'<td><a href="{row["url"]}">{row["pollster"]}</a> ({numeric_grade})</td>')
            lines.append(
                f"<td>{row['start_date'].strftime('%m/%d')}-<wbr>{row['end_date'].strftime('%m/%d')}</td>")
            lines.append(f"<td>{row['harris_pct']:.0f}% : {row['trump_v_harris_pct']:.0f}%</td>")
        else:
            lines.append(f'<td><em>{row["pollster"]}</em></td>')
            lines.append(f"<td></td>")
            lines.append(f"<td></td>")
        lines.append(f"<td>{row['harris_frac']*100:.1f}</td>")

        lines.append("</tr>")
    lines.append("</tbody>")
    lines.append("<tfoot>")
    lines.append("<tr>")
    lines.append(f"<th>Sum {round(df['custom_weight'].sum(), 1)}</th>")
    lines.append("<th>Total</th>")
    lines.append("<th></th>")
    lines.append("<th></th>")
    lines.append(f"<th>Avg {round((df['custom_weight'] * df['harris_frac']).sum() / df['custom_weight'].sum() * 100, 1)}</th>")
    lines.append("</tr>")
    lines.append("</tfoot>")
    lines.append("</table>")
    return '\n'.join(lines)


if __name__ == "__main__":
    print(harris_only_table_html("Georgia"))
