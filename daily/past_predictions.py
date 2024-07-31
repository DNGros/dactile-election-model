import pandas as pd
import requests
from joblib import Memory
from pathlib import Path

from whitmer.plotting import interpolate_color

cur_path = Path(__file__).parent.absolute()

cache = Memory(cur_path / "cache", verbose=0)

@cache.cache
def read_past_vars(
    date: pd.Timestamp,
):
    base_url = "http://dactile.net/p/election-model-archive/"
    url = base_url + date.strftime("%Y-%m-%d") + "/article_vars.json"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def read_all_past_vars(
    start_date: pd.Timestamp = pd.Timestamp("2024-07-26"),
    end_date: pd.Timestamp = pd.Timestamp.now().normalize() - pd.Timedelta(days=1),
) -> dict[str, dict]:
    date_to_vars = {}
    for date in pd.date_range(start_date, end_date):
        vars = read_past_vars(date)
        date_to_vars[date.strftime("%Y-%m-%d")] = vars
    return date_to_vars


def read_all_top_lines():
    all_vars = read_all_past_vars()
    return {
        date: vars["top_line_prob"]
        for date, vars in all_vars.items()
    }


from datetime import datetime, timedelta

def generate_html_calendar(
    data_dict,
    today_value: float,
    today_date: pd.Timestamp = pd.Timestamp.now().normalize(),
):
    data_dict = dict(data_dict)
    data_dict[today_date.strftime("%Y-%m-%d")] = today_value
    # Initialize the HTML string
    html = '<table style="border-collapse: collapse;">'

    events = {
        "2024-07-21": ("👴🏻", "Biden Drops Out"),
        "2024-11-05": ("🗳️", "Election Day"),
    }
    # Set start and end dates
    start_date = datetime.strptime("2024-07-21", "%Y-%m-%d")
    election_day = datetime.strptime("2024-11-05", "%Y-%m-%d")
    end_date = election_day

    # Calculate the number of columns needed (7 days per column)
    total_days = (end_date - start_date).days + 1
    num_columns = (total_days + 6) // 7  # Ceiling division by 7
    num_rows = 7

    # Make a header row with the months
    html += '<tr>'
    month_start_cols = []
    month_names = []
    current_date = start_date
    for col in range(num_columns):
        start_date_of_column = start_date + timedelta(days=(col * num_rows))
        end_date_of_column = start_date + timedelta(days=(col * num_rows + num_rows - 1))
        # If this is the first column or the 1st is in the col then it is a new month
        if col == 0 or end_date_of_column.day <= 7:
            month_start_cols.append(col)
            month_names.append(end_date_of_column.strftime('%b'))
    for col in range(num_columns):
        is_month_start = col in month_start_cols
        if not is_month_start:
            continue
        month_index = month_start_cols.index(col)
        if month_index == len(month_names) - 1:
            month_col_width = 1
        else:
            month_col_width = month_start_cols[month_index + 1] - col
        html += f'<td colspan="{month_col_width}" style="text-align: left; font-size: 12px; font-weight: bold;">{month_names[month_index]}</td>'


    # Generate columns
    for row in range(num_rows):
        html += '<tr>'

        # Generate 7 days for each column (or less for the last column)
        for col in range(num_columns):
            current_date = start_date + timedelta(days=(col * num_rows + row))
            if current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                value = data_dict.get(date_str, None)

                # Define color based on the value
                color = '#ebedf0'  # default color
                if value is not None:
                    value = float(value)
                    color = interpolate_color(fraction=value/100)

                day_val = str(current_date.day)
                # Check to see if there is an event on this day
                if current_date.strftime("%Y-%m-%d") in events:
                    event_emoji, event_text = events[current_date.strftime("%Y-%m-%d")]
                    day_val += f"{event_emoji}"

                value_str = f"{round(value)}%" if value is not None else ''
                if value and date_str != today_date.strftime("%Y-%m-%d"):
                    url = f"http://dactile.net/p/election-model-archive/{date_str}/article.html"
                else:
                    url = None

                cell_value = f'''
                    <td style="width: 30px; height: 30px; border: 1px solid #e1e4e8; background-color: {color};">
                '''
                if url:
                    cell_value += f'<a href="{url}">'
                cell_value += f'''
                        <div style="font-size: 8px; text-align: center; min-width: 30px; text-decoration: none; color: inherit;">{day_val}</div>
                        <div style="font-size: 10px; text-align: center;">{value_str}</div>
                '''
                if url:
                    cell_value += '</a>'
                cell_value += '</td>'
                html += cell_value
                current_date += timedelta(days=1)
            else:
                html += '<td style="width: 30px; height: 30px;"></td>'

        html += '</tr>'

    html += '</table>'
    return html


if __name__ == "__main__":
    # Example data (you would replace this with your actual data)
    data = read_all_top_lines()
    calendar_html = generate_html_calendar(data, today_value=55)
    toy_html = ["<html><head><meta charset='UTF-8'><title>Toy Calendar</title></head><body>"]
    toy_html.append(calendar_html)
    toy_html.append("</body></html>")
    with open("seecalendar.html", "w") as f:
        f.write('\n'.join(toy_html))
    print(calendar_html)