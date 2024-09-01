import functools
import statsmodels.stats.correlation_tools
import scipy.integrate as integrate
from scipy import stats
import gzip
import random
from pathlib import Path
import requests
import json

from graphviz._compat import Literal
from pandas._libs.parsers import defaultdict
from joblib import Memory
import pandas as pd
import numpy as np
import math


cur_path = Path(__file__).parent.absolute()
cache = Memory(cur_path / "cache", verbose=1)


@functools.cache
@cache.cache()
def load_five_thirty_eight_correlation():
    #url = "https://roadtolarissa.com/data/forecast-correlation/pairs-538.json"
    #response = requests.get(url)
    #data = json.loads(response.text)
    p = cur_path / "correlation_data/pairs-538.json.gz"
    data = json.loads(gzip.decompress(p.read_bytes()))
    return _proc_correlations(data)


@functools.cache
@cache.cache()
def load_economist_correlations():
    #url = "https://roadtolarissa.com/data/forecast-correlation/pairs-eco.json"
    #response = requests.get(url)
    #data = json.loads(response.text)
    p = cur_path / "correlation_data/pairs-eco.json.gz"
    data = json.loads(gzip.decompress(p.read_bytes()))
    return _proc_correlations(data)


def load_blended_correlations(
    weight538: float = 0.5,
    weightEconomist: float = 0.5
):
    correlations538 = load_five_thirty_eight_correlation()
    correlationsEconomist = load_economist_correlations()
    correlations = defaultdict(dict)
    for state in correlations538:
        for state2 in correlations538[state]:
            correlations[state][state2] = weight538 * correlations538[state][state2] + weightEconomist * correlationsEconomist[state][state2]
    return correlations


def load_random_correlations():
    return random.choice([load_five_thirty_eight_correlation(), load_economist_correlations()])


def _proc_correlations(data):
    correlations = defaultdict(dict)
    for value in data:
        state1 = value['strA']
        state2 = value['strB']
        correlation = value['cor']
        correlations[state1][state2] = correlation
    return correlations


@cache.cache()
def _get_map_buf(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download {url}")
    return response.content


def _parse_buf_to_dataframe(buf):
    """map data parsing"""
    states = [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
        "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
        "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
        "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    ]

    int16_array = np.frombuffer(buf, dtype=np.int16)
    assert len(int16_array) == 40000 * len(states)
    reshaped_array = int16_array.reshape((40000, len(states)))
    percentage_array = reshaped_array / 10000

    df = pd.DataFrame(percentage_array, columns=states)

    return df


@functools.cache
def _get_538_map_df():
    buf = gzip.decompress((cur_path / "correlation_data/maps-538.buf.gz").read_bytes())
    return _parse_buf_to_dataframe(buf)


def _get_eco_map_df():
    buf = gzip.decompress((cur_path / "correlation_data/maps-eco.buf.gz").read_bytes())
    return _parse_buf_to_dataframe(buf)


def _find_buf_df_correlations(df):
    correlations = defaultdict(dict)
    for state1 in df.columns:
        for state2 in df.columns:
            correlations[state1][state2] = df[state1].corr(df[state2])
    return correlations


def _find_buf_df_covariances(df):
    covariances = defaultdict(dict)
    for state1 in df.columns:
        for state2 in df.columns:
            covariances[state1][state2] = df[state1].cov(df[state2])
    return covariances


def apply_correlation(
    state: str,
    value: float,
    correlation_power: float = 1.0
):
    correlations = load_random_correlations()
    vals = {}
    for state2, correlation in correlations[state].items():
        vals[state2] = value * (correlation**correlation_power)
    return vals


@functools.cache
def load_538_covariances():
    df = _get_538_map_df()
    return _find_buf_df_covariances(df)


@functools.cache
def load_eco_covariances():
    df = _get_eco_map_df()
    return _find_buf_df_covariances(df)


def load_random_covariances():
    if random.choice([True, False]):
        return load_538_covariances()
    else:
        return load_eco_covariances()


@functools.cache
def calc_scale_factor_for_t_dist(degrees, target_average_difference, loc=0.0):
    """I am not sure the analytical solution so we can figure it out over the pdf"""
    t_pdf = lambda x: stats.t.pdf(x, degrees, loc=loc)

    # Compute the expected absolute deviation from the mean
    expected_absolute_deviation, _ = integrate.quad(lambda x: np.abs(x) * t_pdf(x), -np.inf, np.inf)

    #print(f"Expected absolute deviation for t-distribution with df={degrees}: {expected_absolute_deviation}")

    # Calculate scale factor
    scale_factor = target_average_difference / expected_absolute_deviation

    return scale_factor


@functools.cache
def calculate_t_percentile_move(
    degrees,
    target_average_difference,
    percentile
):
    scale = calc_scale_factor_for_t_dist(degrees, target_average_difference)
    dist = stats.t(df=degrees, loc=0, scale=scale)
    # Probably some analytical solution but we can just sample
    sample = dist.sample(1_000_000)
    return np.percentile(np.abs(sample), percentile)



@functools.cache
def get_multivariate_normal_dist(
    source: Literal['538', 'eco'],
    std_dev: float,
    states: list[str] = None
):
    if source == '538':
        #covariance_dict = load_538_covariances()
        correlation_dict = load_five_thirty_eight_correlation()
    elif source == 'eco':
        #covariance_dict = load_eco_covariances()
        correlation_dict = load_economist_correlations()
    else:
        raise ValueError("Invalid")
    if not states:
        states = list(correlation_dict.keys())
    else:
        correlation_dict = {
            state: {
                state2: correlation_dict[state][state2] for state2 in states
            } for state in states
        }
    correlation_matrix = np.array(
        [[correlation_dict[state1][state2] for state2 in states] for state1 in states])
    dist = stats.multivariate_normal(
        cov=corr2cov(correlation_matrix, std_dev), mean=np.zeros(len(states)))
    return dist, states


def correlation_dict_to_matrix(correlation_dict):
    states = list(correlation_dict.keys())
    return (
        np.array(
            [[correlation_dict[state1][state2] for state2 in states] for state1 in states]),
        states
    )


@functools.cache
def get_multivariate_t_dist(
    source: Literal['538', 'eco'],
    degrees_freedom: float,
    states: list[str] = None
):
    """Gets a multivariate t-distribution for state changes from extracted
    correlations"""
    if source == '538':
        correlation_dict = load_five_thirty_eight_correlation()
    elif source == 'eco':
        correlation_dict = load_economist_correlations()
    else:
        raise ValueError("Invalid")
    # Filter to states needed
    if not states:
        states = list(correlation_dict.keys())
    else:
        correlation_dict = {
            state: {
                state2: correlation_dict[state][state2] for state2 in states
            } for state in states
        }
    # Calculate the covariance matrix
    correlation_matrix = np.array(
        [[correlation_dict[state1][state2] for state2 in states] for state1 in states])
    dist = stats.multivariate_t(
        df=degrees_freedom,
        shape=corr2cov(correlation_matrix, 1.0),
        loc=np.zeros(len(states)),
    )
    return dist, states


def get_random_multivariate_normal_dist(std_dev=1.0, states=None):
    if random.choice([True, False]):
        return get_multivariate_normal_dist('538', std_dev, states)
    else:
        return get_multivariate_normal_dist('eco', std_dev, states)


def get_random_multivariate_t_dist(degrees_freedom, states=None):
    """Selects a correlated t-dist from one of our available sources"""
    if random.choice([True, False]):
        return get_multivariate_t_dist('538', degrees_freedom, states)
    else:
        return get_multivariate_t_dist('eco', degrees_freedom, states)


def corr2cov(corr, std):
    """
    convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.
    """
    # https://github.com/statsmodels/statsmodels/blob/bc1899510adacebf1ec351e92a78b7421c17dbb0/statsmodels/stats/moment_helpers.py#L259-L284
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov


def get_correlation_matrix_pow_for_one_state(
    source: Literal['538', 'eco'],
    state: str,
    power: float,
    states: list[str] = None
):
    """Here we want to get a version of a correlation matrix, but we just
    want the correlations to one state raised to some power"""
    if source == '538':
        correlation_dict = load_five_thirty_eight_correlation()
    elif source == 'eco':
        correlation_dict = load_economist_correlations()
    else:
        raise ValueError("Invalid")
    if states:
        correlation_dict = {
            state: {
                state2: correlation_dict[state][state2] for state2 in states
            } for state in states
        }
    matrix, n_states = correlation_dict_to_matrix(correlation_dict)
    assert states == n_states
    # Now make a new version with the given state powered
    state_index = states.index(state)
    # start with identity matrix
    v = np.eye(matrix.shape[0])
    # Copy a scaled version of the row
    v[state_index] = matrix[state_index] ** power
    # copy col
    v[:, state_index] = matrix[:, state_index] ** power
    if not is_positive_semidefinite(v):
        v = statsmodels.stats.correlation_tools.cov_nearest(v, threshold=1e-6)
    assert is_positive_semidefinite(v)
    assert np.allclose(v, v.T)
    return v, states


def is_positive_semidefinite(M):
    eigenvalues = np.linalg.eigvals(M)
    return np.all(eigenvalues >= 0)


if __name__ == "__main__":
    loc = 0.00
    df = 5
    scale = calc_scale_factor_for_t_dist(df, 0.05, loc=loc)
    dist = stats.t(df=df, loc=loc, scale=scale)
    print(f"found {scale=}")
    print(dist.mean())
    samples = dist.rvs(size=1000000)
    print(np.mean(np.abs(samples)))
    exit()
    covariance_dict = load_538_covariances()
    print(np.sqrt(covariance_dict['CA']['CA']))
    correlation_dict = load_five_thirty_eight_correlation()
    correlation_matrix = np.array(
        [[correlation_dict[state1][state2] for state2 in correlation_dict.keys()] for state1 in correlation_dict.keys()])
    std_dev = 2
    D = np.diag([std_dev] * len(correlation_dict))
    new_covariance_matrix = D @ correlation_matrix @ D
    old_covariance_matrix = np.array(
        [[covariance_dict[state1][state2] for state2 in covariance_dict.keys()] for state1 in covariance_dict.keys()])
    print("old matrix")
    print(old_covariance_matrix)
    # divide the old covariance matrix by its diagonal
    old_covariance_matrix /= np.sqrt(np.diag(old_covariance_matrix))
    print("attempt at renorm")
    print(old_covariance_matrix)
    print("covar from correlation")
    print(new_covariance_matrix)
    print("correlation")
    print(correlation_matrix)
    print(corr2cov(correlation_matrix, std_dev))
    exit()
    print(load_five_thirty_eight_correlation())
    print(apply_correlation("MI", 0.02))