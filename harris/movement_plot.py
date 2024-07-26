from harris.margins import get_margins_df
from harris.typical_variances import get_margins_df_custom_avg, make_state_movements_plot

if __name__ == "__main__":
    df = get_margins_df_custom_avg(True)
    make_state_movements_plot(
        get_margins_df_custom_avg(True),
        #get_margins_df(True)
    )
