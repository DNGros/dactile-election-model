import pandas as pd
import math
import numpy as np

from scipy import stats

import simulate
from county_model import get_model_diff_stats_pres_to_gov
from election_statics import BIDEN, WHITMER
from historical_elections import president_county_data, format_mit_county_data_into_simple_form, \
    michigan_county_governor_2018, format_cnn_county_data_simple_form, michigan_county_governor_2022
from plotting import interpolate_color
from simulate import estimate_fracs, simulate_election_mc


def make_whitmer_elect_table():
    df = president_county_data(year=2020)
    df = df[df["state_code"] == "MI"]
    df_pres_2020 = format_mit_county_data_into_simple_form(df)
    df2016 = president_county_data(year=2016)
    df_pres_2016 = format_mit_county_data_into_simple_form(
        df2016[df2016["state_code"] == "MI"]
    )
    print("Pres 2016")
    print(df_pres_2016)
    df = michigan_county_governor_2018()
    df_gov_2018 = format_cnn_county_data_simple_form(df)
    df = michigan_county_governor_2022()
    df_gov_2022 = format_cnn_county_data_simple_form(df)

    # Make a df with the dem votes
    def df_to_row(election, candidates, df):
        return {
            "Election": election,
            "Candidates": candidates,
            "D Votes": df["candidatevotes_DEM"].sum(),
            "R Votes": df["candidatevotes_REP"].sum(),
            "D %": df["candidatevotes_DEM"].sum() / (df["candidatevotes_DEM"].sum() + df["candidatevotes_REP"].sum()) * 100,
            "R %": df["candidatevotes_REP"].sum() / (df["candidatevotes_DEM"].sum() + df["candidatevotes_REP"].sum()) * 100,
        }
    tdf = pd.DataFrame(
        data=[
            df_to_row("Pres 2016", "Clinton vs Trump", df_pres_2016),
            df_to_row("Gov 2018", "Whitmer vs Schuette", df_gov_2018),
            df_to_row("Pres 2020", "Biden vs Trump", df_pres_2020),
            df_to_row("Gov 2022", "Whitmer vs Dixon", df_gov_2022),
        ]
    )
    print(tdf)
    return tdf.to_html(formatters={"D %": "{:.2f}".format, "R %": "{:.1f}".format}, index=False, border=0)


def shift_table():
    df = president_county_data(year=2020)
    df = df[df["state_code"] == "MI"]
    df_pres_2020 = format_mit_county_data_into_simple_form(df)
    df2016 = president_county_data(year=2016)
    df_pres_2016 = format_mit_county_data_into_simple_form(
        df2016[df2016["state_code"] == "MI"]
    )
    print("Pres 2016")
    print(df_pres_2016)
    df = michigan_county_governor_2018()
    df_gov_2018 = format_cnn_county_data_simple_form(df)
    df = michigan_county_governor_2022()
    df_gov_2022 = format_cnn_county_data_simple_form(df)

    def calculate_margin_of_error(scale, sample_size, confidence_level=0.90):
        # Degrees of freedom
        df = sample_size - 1
        # Critical value from the t-distribution
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
        # Standard error
        standard_error = scale / np.sqrt(sample_size)
        # Margin of error
        margin_of_error = t_critical * standard_error
        return margin_of_error

    def make_row(name, mean, variance):
        mean_percent = mean * 100
        sd = variance ** 0.5 * 100
        ci_90 = 1.645 * sd * 100

        return {
            "Change": name,
            "Mean Dem Points": f"{mean_percent:.1f}",
            #"Variance": f"{variance_percent:.2f}",
            "County-level σ": f"{sd:.2f}",
            #"T 90": f"{calculate_margin_of_error(variance, 6) * 100:.4f}",
        }
    rows = []
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2016, df_gov_2018, weighting_cat="pres")
    rows.append(make_row("Pres 2016 → Gov 2018", mean, variance))
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2020, df_gov_2018, weighting_cat="pres")
    rows.append(make_row("Pres 2020 → Gov 2018", mean, variance))
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2016, df_gov_2022, weighting_cat="gov")
    rows.append(make_row("Pres 2016 → Gov 2022", mean, variance))
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2020, df_gov_2022, weighting_cat="gov")
    rows.append(make_row("Pres 2020 → Gov 2022", mean, variance))
    #mean, variance = get_model_diff_stats_pres_to_gov(df_gov_2018, df_gov_2022, weighting_cat="gov")
    #rows.append(make_row("Gov 2018 -> Gov 2022", mean, variance))
    tdf = pd.DataFrame(data=rows)
    print(tdf)
    return tdf.to_html(index=False, border=0)

import concurrent.futures

default_params = {
    "dem_candidate": WHITMER,
    "poll_miss": "adjusted",
}


def compute_win_chance(mean, c90, correlation_power):
    std_dev = c90 / 1.645
    params = default_params.copy()
    params["chaos_dem_mean"] = mean / 100
    params["chaos_std_dev"] = std_dev / 100
    params["correlation_power"] = correlation_power
    win_chance, _ = estimate_fracs(simulate_election_mc(**params))
    return (mean, c90, win_chance)


means = [-8, -6, -4, -3, -2, -1, 0, 1, 2]
c90s = [0, 1, 2, 4, 6, 8]


def get_variance_table_dict_results(correlation_power=1.0):
    base_win_chance, _ = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss="default",
        correlation_power=correlation_power,
    ))
    adjust_win_chance, _ = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss="adjusted",
        correlation_power=correlation_power,
    ))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            (mean, c90): executor.submit(compute_win_chance, mean, c90, correlation_power)
            for c90 in c90s
            for mean in means
        }
        win_chances = {
            (mean, c90): (future.result()[2], future.result()[2] > base_win_chance, future.result()[2] > adjust_win_chance)
            for (mean, c90), future in futures.items()
        }
    return win_chances


def make_variance_table(correlation_power=1.0):
    win_chances = get_variance_table_dict_results(correlation_power)
    lines = ["<table>"]
    lines.append("<tr>")
    lines.append("<th>Mean Adjustment → / C90 ↓ </th>")
    for mean in means:
        lines.append(f"<th>{mean}</th>")
    lines.append("</tr>")

    for c90 in c90s:
        lines.append("<tr>")
        lines.append(f"<td>±{c90}</td>")
        for mean in means:
            win_chance, is_above_base, is_above_adjust = win_chances[(mean, c90)]
            color = interpolate_color(fraction=win_chance, alpha=0.2)
            if is_above_adjust:
                emoji = "🟢"
            elif is_above_base:
                emoji = "🟡"
            else:
                emoji = "🔴"
            lines.append(f"<td style='background-color:{color}'>{win_chance * 100:.1f}% {emoji}</td>")
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def alt_correlation_table():
    powers = [1.0, 2.0, 3.0, 4.0, 999]
    lines = ["<table>"]
    lines.append("<tr>")
    # make a header rows with the means
    lines.append("<th>Correlation Change</th>")
    lines.append("<th>D Win Prob</th>")
    lines.append("<th>🟢-frontier</th>")
    lines.append("</tr>")

    for power in powers:
        lines.append("<tr>")
        v = f"correlation^{power}"
        if power > 100:
            v += " (just MI)"
        lines.append(f"<td>{v}</th>")
        win_chance, _ = estimate_fracs(simulate_election_mc(
            dem_candidate=WHITMER,
            poll_miss="adjusted",
            correlation_power=power,
        ))
        lines.append(f"<td>{win_chance * 100:.0f}%</td>")
        # Figure out frontier
        win_chances = get_variance_table_dict_results(power)
        frontier = []
        for ci in c90s:
            best = None
            for mean in means:
                win_chance, is_above_base, is_above_adjust = win_chances[(mean, ci)]
                if is_above_adjust:
                    best = (mean, ci)
                    break
            if best is not None:
                frontier.append(f"{best[0]}±{best[1]}")
        lines.append(f"<td>{','.join(frontier)}</td>")
        lines.append("</tr>")
    # Add a horizontal line and the base case
    lines.append("<tr>")
    lines.append("<td>No Adjustment (Biden)</th>")
    win_chance, _ = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss="adjusted",
    ))
    lines.append(f"<td>{win_chance * 100:.1f}%</td>")
    lines.append("</table>")
    return "\n".join(lines)


def openlabs_table():
    return """<table>
  <thead>
    <tr>
      <th rowspan="2">State</th>
      <th rowspan="2">Biden</th>
      <th colspan="3">Two-way horserace</th>
      <th colspan="3">Name-recognition adjusted</th>
      <th colspan="2">Our Estimates: Mean {538, Economist}</th>
    </tr>
    <tr>
      <th>Whitmer</th>
      <th>Delta</th>
      <th>MI Correlation</th>
      <th>Whitmer</th>
      <th>Delta</th>
      <th>MI Correlation</th>
      <th>power=1</th>
      <th>power=3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>WI</td>
      <td>49.3%</td>
      <td>51.7%</td>
      <td>2.4%</td>
      <td>0.39</td>
      <td>53.1%</td>
      <td>3.8%</td>
      <td>0.61</td>
      <td>0.84 {0.82, 0.85}</td>
      <td>0.59 {0.56, 0.62}</td>
    </tr>
    <tr>
      <td>NE-2</td>
      <td>48.8%</td>
      <td>51.4%</td>
      <td>2.6%</td>
      <td>0.43</td>
      <td>53.3%</td>
      <td>4.5%</td>
      <td>0.73</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>PA</td>
      <td>47.6%</td>
      <td>49.7%</td>
      <td>2.1%</td>
      <td>0.34</td>
      <td>51.1%</td>
      <td>3.5%</td>
      <td>0.56</td>
      <td>0.84 {0.81, 0.86}</td>
      <td>0.59 {0.54, 0.64}</td>
    </tr>
    <tr>
      <td>MI</td>
      <td>47.5%</td>
      <td>53.6%</td>
      <td>6.1%</td>
      <td>1.00</td>
      <td>53.7%</td>
      <td>6.2%</td>
      <td>1.00</td>
      <td>1.00 {1.00, 1.00}</td>
      <td>1.00 {1.00, 1.00}</td>
    </tr>
    <tr>
      <td>NV</td>
      <td>46.6%</td>
      <td>48.3%</td>
      <td>1.7%</td>
      <td>0.28</td>
      <td>51.7%</td>
      <td>5.1%</td>
      <td>0.82</td>
      <td>0.57 {0.69, 0.45}</td>
      <td>0.21 {0.33, 0.09}</td>
    </tr>
    <tr>
      <td>GA</td>
      <td>46.3%</td>
      <td>47.5%</td>
      <td>1.2%</td>
      <td>0.20</td>
      <td>49.7%</td>
      <td>3.4%</td>
      <td>0.55</td>
      <td>0.53 {0.55, 0.52}</td>
      <td>0.15 {0.16, 0.14}</td>
    </tr>
    <tr>
      <td>AZ</td>
      <td>46.1%</td>
      <td>47.8%</td>
      <td>1.7%</td>
      <td>0.28</td>
      <td>49.7%</td>
      <td>3.6%</td>
      <td>0.58</td>
      <td>0.56 {0.61, 0.51}</td>
      <td>0.18 {0.22, 0.13}</td>
    </tr>
    <tr>
      <td>NC</td>
      <td>45.9%</td>
      <td>48.1%</td>
      <td>2.2%</td>
      <td>0.36</td>
      <td>49.7%</td>
      <td>3.8%</td>
      <td>0.61</td>
      <td>0.70 {0.66, 0.73}</td>
      <td>0.34 {0.29, 0.40}</td>
    </tr>
  </tbody>
</table>"""


if __name__ == "__main__":
    #make_whitmer_elect_table()
    #shift_table()
    print(make_variance_table())
    #print(alt_correlation_table())
    print("Done")
