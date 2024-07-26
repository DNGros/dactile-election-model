from scipy import stats
import numpy as np
from sample_tools import sample_one_of_distributions
import pandas as pd

default_poll_miss_degrees_freedom = 5

# make a t distribution with 5 degrees of freedom that can sample from
default_poll_miss = stats.t(df=default_poll_miss_degrees_freedom, scale=0.03)
"""
Models the error of a poll to the actual result

From https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=110867585
the average poll miss in a competitive state is about 2%.

They chose to model this with a t-distribution with 5 degrees of freedom.
These were fit from their historical model.

We tune the scale to get this to be ~2%.
"""
#adjusted_poll_miss = stats.t(df=5, scale=0.061)  # Orig
adjusted_poll_miss = stats.t(df=5, scale=0.057)
poll_miss_div = 1
poll_miss_for_other_mis_div = 4

whitmer_mi_bump = sample_one_of_distributions([
    #stats.norm(loc=0.051420, scale=np.sqrt(0.0004156)),  # Pres 2016 -> Gov 2018
    #stats.norm(loc=0.033255, scale=np.sqrt(0.0007264)),  # Pres 2020 -> Gov 2018
    #stats.norm(loc=0.060528, scale=np.sqrt(0.0004615)),  # Pres 2016 -> Gov 2022
    #stats.norm(loc=0.042059, scale=np.sqrt(0.0002676)),  # Pres 2020 -> Gov 2022

    stats.norm(loc=0.051420, scale=np.sqrt(0.0004156)/9.1104335791443),  # Pres 2016 -> Gov 2018
    stats.norm(loc=0.033255, scale=np.sqrt(0.0007264)/9.1104335791443),  # Pres 2020 -> Gov 2018
    stats.norm(loc=0.060528, scale=np.sqrt(0.0004615)/9.1104335791443),  # Pres 2016 -> Gov 2022
    stats.norm(loc=0.042059, scale=np.sqrt(0.0002676)/9.1104335791443),  # Pres 2020 -> Gov 2022

    #stats.norm(loc=0.051420, scale=0),  # Pres 2016 -> Gov 2018
    #stats.norm(loc=0.033255, scale=0),  # Pres 2020 -> Gov 2018
    #stats.norm(loc=0.060528, scale=0),  # Pres 2016 -> Gov 2022
    #stats.norm(loc=0.042059, scale=0),  # Pres 2020 -> Gov 2022
])

#chaos_factor = stats.truncnorm(-0.5, 0, loc=-0.02, scale=0.01)
#chaos_factor = stats.norm(loc=-0.04, scale=0.02)
default_chaos_factor = stats.norm(loc=-0.00, scale=0.0002)

#harris_national_change = stats.t(loc=-0.00, scale=0.012, df=2)
harris_national_change = stats.norm(loc=-0.0000534, scale=0.02/np.sqrt(2/np.pi))

default_movement_degrees_freedom = 5
default_movement_cur_cycle_average_multiple = 1

harris_article_calc_date = pd.Timestamp("2024-07-21")
harris_delta_error_default = 0.035


polling_measurable_error_frac = 0.5
"""The amount of polling miss that can be explained with polling.
The rest is assumed unknowable (due just general flaws with polling 
for example only reaching certain types of voters or general flaws
in the methodology of dominate polsters)"""
default_poll_time_penalty = 9
"""How much we penalize old polls. Roughly how long it
takes for a poll to have half weight"""
default_scale_national_for_states = 1
"""How much we scale the r^2 of the state"""
dropout_day = pd.to_datetime('2024-07-21')
"""When Biden dropped out"""
harris_start_display_date = pd.to_datetime('2024-07-12')

swing_states = ['Pennsylvania', 'Wisconsin', 'Georgia', 'Michigan', 'North Carolina', 'Arizona', 'Nevada',
                'Florida']
