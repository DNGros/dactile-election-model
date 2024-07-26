import pandas as pd
import plotly.colors as pc
from plotly import graph_objects as go
import plotly.io as pio

from election_statics import BIDEN, get_electoral_votes, WHITMER
from simulate import estimate_fracs, simulate_election_mc
from pathlib import Path

from state_correlations import load_blended_correlations

cur_path = Path(__file__).parent.absolute()
map_base = cur_path / 'articlegen/imgs/maps'
map_base.mkdir(exist_ok=True, parents=True)

colorscale = [
    [0, 'rgb(178,24,43)'],  # Deep red for low Democratic win fraction
    [0.5, 'rgb(255,255,255)'],  # White for 50/50
    [1, 'rgb(33,102,172)']  # Deep blue for high Democratic win fraction
]


def plot_election_map(
    overall_frac,
    dem_win_fracs,
    title: str = None,
    save_path=None,
    show=None,
    candidate_name: str = "Dem"
):
    # Create a DataFrame with state codes and Democratic win fractions
    _, state_to_votes = get_electoral_votes()
    df = pd.DataFrame(list(dem_win_fracs.items()), columns=['state', 'dem_win_frac'])

    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=df['state'],
        z=df['dem_win_frac'],
        locationmode='USA-states',
        colorscale=colorscale,
        showscale=False  # Remove the color bar legend
    ))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })


    overall_color = interpolate_color(
        colorscale,
        overall_frac,
        alpha=0.9
    )

    opaque_color = interpolate_color(colorscale, overall_frac)
    r, g, b = map(int, opaque_color.strip('rgb()').split(','))
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    text_color = 'black' if brightness > 128 else 'white'

    # Add overall fraction as text annotation
    fig.add_annotation(
        text=f"{candidate_name} Total Win Prob: {overall_frac*100:.0f}%",
        xref="paper",
        yref="paper",
        x=0.5, y=0.5,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=12, color=text_color),
        bgcolor=overall_color,
        bordercolor="black",
        borderwidth=2,
        borderpad=4
    )

    # Update the layout
    fig.update_layout(
        title_text=title,
        geo=dict(
            scope='usa',
            showlakes=False,  # Remove lakes for a cleaner look
            landcolor='rgb(240,240,240,0)',  # Light grey land
        ),
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),  # Reduce margins
        dragmode=False,  # Disable dragging/panning
        autosize=True,  # Allow the figure to autosize
    )

    # Disable zoom and other controls
    fig.update_layout(
        updatemenus=[],  # Remove the controls toolbar
        showlegend=False,
    )

    # Disable all axes
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)

    if save_path:
        pio.write_html(
            fig, save_path, auto_open=show, config={'displayModeBar': False},
            include_plotlyjs='cdn',
        )

    # Show the plot
    if show and not save_path:
        fig.show(config={'displayModeBar': False})
    return save_path



import numpy as np

def interpolate_color(colorscale=colorscale, fraction=0, alpha=None):
    if fraction <= 0:
        color = colorscale[0][1]
    elif fraction >= 1:
        color = colorscale[-1][1]
    else:
        for i in range(len(colorscale) - 1):
            lower, upper = colorscale[i], colorscale[i+1]
            if lower[0] <= fraction <= upper[0]:
                t = (fraction - lower[0]) / (upper[0] - lower[0])
                lower_rgb = np.array([int(lower[1].strip('rgb()').split(',')[j]) for j in range(3)])
                upper_rgb = np.array([int(upper[1].strip('rgb()').split(',')[j]) for j in range(3)])
                rgb = lower_rgb + t * (upper_rgb - lower_rgb)
                color = f'rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})'
                break
        else:
            color = colorscale[-1][1]  # Fallback to last color if something goes wrong

    if alpha is not None:
        # Convert rgb to rgba
        rgb_values = color.strip('rgb()').split(',')
        return f'rgba({",".join(rgb_values)},{alpha})'
    else:
        return color


def make_all_plots():
    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss="default",
    ))
    show = False
    p = plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'biden_base.html'),
        show=show,
        title="If the election was today (polling avg on July 6th):",
        candidate_name="Biden",
    )
    print("Saved to", p)
    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=BIDEN,
        poll_miss="adjusted",
    ))
    p = plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'biden_adjusted.html'),
        show=show,
        title="Adjusted variance:",
        candidate_name="Biden",
    )
    print("Saved to", p)
    overall_frac, state_frac = estimate_fracs(simulate_election_mc(
        dem_candidate=WHITMER,
        poll_miss="adjusted",
    ))
    p = plot_election_map(
        overall_frac, state_frac,
        save_path=str(map_base / 'whitmer_adjusted.html'),
        show=show,
        title="Applying Whitmer Factor to Each State:",
        candidate_name="Whitmer",
    )
    print("Saved to", p)
    p = plot_state_correlations(
        save_path=str(map_base / 'state_correlations.html'),
    )
    print("Saved to", p)


def plot_state_correlations(save_path= None):
    state_to_correlation = load_blended_correlations()
    state_to_correlation = state_to_correlation['MI']
    # Make a map of the correlations
    df = pd.DataFrame(list(state_to_correlation.items()), columns=['state', 'MI Correlation'])
    # Create a custom green color scale
    green_colorscale = [
        [0, 'rgb(255,255,255)'],  # Lightest green
        [0.25, 'rgb(200,245,230)'],  # Lightest green
        #[0.5, 'rgb(116,196,118)'],
        #[0.75, 'rgb(35,139,69)'],
        [1, 'rgb(0,90,50)']  # Darkest green
    ]

    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=df['state'],
        z=df['MI Correlation'],
        locationmode='USA-states',
        colorscale=green_colorscale,
        showscale=False  # Show the color bar legend
    ))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    # Update the layout
    fig.update_layout(
        title_text="Correlation with MI",
        geo=dict(
            scope='usa',
            showlakes=False,  # Remove lakes for a cleaner look
            landcolor='rgb(240,240,240,0)',  # Light grey land
        ),
        margin=dict(l=0, r=0, t=30, b=0),  # Reduce margins
        dragmode=False,  # Disable dragging/panning
        autosize=True,  # Allow the figure to autosize
    )
    # Add text labels for each state
    #for i, row in df.iterrows():
    #    state = row['state']
    #    if state not in [
    #        # Only swing states
    #        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
    #    ]:
    #        continue
    #    fig.add_annotation(
    #        go.layout.Annotation(
    #            text=f"{row['MI Correlation']:.2f}",
    #            x=row['state'],
    #            y=row['state'],
    #            showarrow=False,
    #            font=dict(size=12, color='black'),
    #            xref='x',
    #            yref='y'
    #        )
    #    )

    # Disable zoom and other controls
    fig.update_layout(
        updatemenus=[],  # Remove the controls toolbar
        showlegend=False,
    )

    # Disable all axes
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    if save_path:
        pio.write_html(fig, save_path, auto_open=False, config={'displayModeBar': False})
    #fig.show()
    return save_path


if __name__ == "__main__":
    make_all_plots()
    plot_state_correlations()
    print("Done plotting")


