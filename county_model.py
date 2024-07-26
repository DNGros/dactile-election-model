"""
Model the state county results to predict variance of shift
"""
import math
from typing import Literal


from historical_elections import president_county_data, format_mit_county_data_into_simple_form, \
    michigan_county_governor_2018, format_cnn_county_data_simple_form, michigan_county_governor_2022
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def get_model_diff_stats_pres_to_gov(
    df_pres,
    df_gov,
    weighting_cat: Literal["pres", "gov"]
):
    assert set(df_pres.columns) == set(df_gov.columns)
    assert len(df_pres) == len(df_gov)
    df_pres = df_pres.copy()
    df_gov = df_gov.copy()

    def main_party_votes(row):
        return row["candidatevotes_DEM"] + row["candidatevotes_REP"]
    def dem_frac(row):
        return row["candidatevotes_DEM"] / (row["candidatevotes_DEM"] + row["candidatevotes_REP"])

    df_pres["main_party_votes"] = df_pres.apply(main_party_votes, axis=1)
    df_gov["main_party_votes"] = df_gov.apply(main_party_votes, axis=1)
    df_pres["dem_frac"] = df_pres.apply(dem_frac, axis=1)
    df_gov["dem_frac"] = df_gov.apply(dem_frac, axis=1)

    # weight each county by all
    def set_frac_of_all_major_party(df):
        df['frac_of_all_major_party'] = df['main_party_votes'] / df['main_party_votes'].sum()
    set_frac_of_all_major_party(df_pres)
    set_frac_of_all_major_party(df_gov)

    # Join the gov and pres data
    df_both = pd.merge(
        df_pres, df_gov,
        on=["state_county"],
        suffixes=("_pres", "_gov"),
        validate="one_to_one"
    )
    df_both['dem_change'] = df_both['dem_frac_gov'] - df_both['dem_frac_pres']
    weight_col = "frac_of_all_major_party_gov"
    # plot the dem change distribution as a kde
    #df_both["dem_change"].plot.kde()

    (mean, variance) = fit_normal(df_both["dem_change"], df_both[weight_col])
    # plot the normal distribution
    #x = np.linspace(-0.1, 0.15, 100)
    #y = t.pdf(x, df=len(df_both) - 1, loc=mean, scale=np.sqrt(variance))
    #plt.plot(x, y, 'r--')
    #plt.show()
    return (mean, variance)


def fit_normal(values, weights):
    # prepare
    values = np.array(values)
    weights = np.array(weights)

    # estimate mean
    weights_sum = weights.sum()
    mean = (values * weights).sum() / weights_sum

    # estimate variance
    errors = (values - mean) ** 2
    variance = (errors * weights).sum() / weights_sum

    return (mean, variance)


if __name__ == "__main__":
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

    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2016, df_gov_2018, weighting_cat="pres")
    count = len(df_gov_2022)
    print(f"stats.norm(loc={mean:.6f}, scale=np.sqrt({variance:.7f})/{math.sqrt(count)}), # Pres 2016 -> Gov 2018")
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2020, df_gov_2018, weighting_cat="pres")
    print(f"stats.norm(loc={mean:.6f}, scale=np.sqrt({variance:.7f})/{math.sqrt(count)}), # Pres 2020 -> Gov 2018")
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2016, df_gov_2022, weighting_cat="gov")
    print(f"stats.norm(loc={mean:.6f}, scale=np.sqrt({variance:.7f})/{math.sqrt(count)}), # Pres 2016 -> Gov 2022")
    mean, variance = get_model_diff_stats_pres_to_gov(df_pres_2020, df_gov_2022, weighting_cat="gov")
    print(f"stats.norm(loc={mean:.6f}, scale=np.sqrt({variance:.7f})/{math.sqrt(count)}), # Pres 2020 -> Gov 2022")
